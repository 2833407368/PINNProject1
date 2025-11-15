import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# =========================
# 1. r_c(α, θ) 神经网络
# =========================
class RCNet(nn.Module):
    def __init__(self, in_dim=2, hidden=64, n_layers=3, min_r=0.1):
        super().__init__()
        self.min_r = min_r
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.model = nn.Sequential(*layers)

        # Xavier 初始化
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pts):
        raw = self.model(pts)
        # softplus + min_r 确保 r_c > 0，防止塌到 0 变成断面
        return self.min_r + torch.nn.functional.softplus(raw)


# =========================
# 2. 附录公式：坐标 + 偏导（有梯度）
# =========================
def compute_xyz_and_partials(rc_net, alpha, theta, params):
    alpha = alpha.to(DEVICE, dtype=DTYPE)
    theta = theta.to(DEVICE, dtype=DTYPE)
    pts = torch.cat([alpha, theta], dim=1)
    pts.requires_grad_(True)

    rc = rc_net(pts)

    # Eq.(7)(8) 中的 r 和 r'
    r0 = params['r0']
    r0p = params['r0p']
    theta0 = params['theta0']
    tanphi = params['tanphi']
    rh = params['rh']
    theta_h = params['theta_h']

    theta_flat = pts[:, 1:2]

    r = r0 * torch.exp((theta_flat - theta0) * tanphi)
    rp = r0p * torch.exp(-(theta_flat - theta0) * tanphi)

    dr_dtheta = r * tanphi
    drp_dtheta = -rp * tanphi

    rsum = r + rp
    drsum_dtheta = dr_dtheta + drp_dtheta

    cos_t = torch.cos(theta_flat)
    sin_t = torch.sin(theta_flat)

    # 中点 (xm, ym, zm)，按附录表达式
    xm = torch.zeros_like(rc)
    ym = 0.5 * rsum * cos_t - rh * math.cos(theta_h)
    zm = rh * math.sin(theta_h) - 0.5 * rsum * sin_t

    # Eq.(9) 中的极坐标到 (x,y,z)
    sin_a = torch.sin(alpha)
    cos_a = torch.cos(alpha)

    x = xm + rc * sin_a
    y = ym + rc * cos_a * cos_t
    z = zm - rc * cos_a * sin_t

    # 自动微分求 rc 对 α、θ 的偏导
    grads = torch.autograd.grad(rc.sum(), pts, create_graph=True)[0]
    drc_dalpha = grads[:, 0:1]
    drc_dtheta = grads[:, 1:2]

    # 链式求导求 ∂x/∂α, ∂x/∂θ 等（附录里的推导）
    dx_dalpha = drc_dalpha * sin_a + rc * cos_a
    dx_dtheta = drc_dtheta * sin_a

    dym_dtheta = 0.5 * drsum_dtheta * cos_t - 0.5 * rsum * sin_t
    dy_dalpha = drc_dalpha * cos_a * cos_t - rc * sin_a * cos_t
    dy_dtheta = dym_dtheta + drc_dtheta * cos_a * cos_t - rc * cos_a * sin_t

    dzm_dtheta = -0.5 * drsum_dtheta * sin_t - 0.5 * rsum * cos_t
    dz_dalpha = -drc_dalpha * cos_a * sin_t + rc * sin_a * sin_t
    dz_dtheta = dzm_dtheta - drc_dtheta * cos_a * sin_t - rc * cos_a * cos_t

    return {
        'pts': pts,
        'rc': rc,
        'x': x, 'y': y, 'z': z,
        'dx_dalpha': dx_dalpha, 'dx_dtheta': dx_dtheta,
        'dy_dalpha': dy_dalpha, 'dy_dtheta': dy_dtheta,
        'dz_dalpha': dz_dalpha, 'dz_dtheta': dz_dtheta
    }


# =========================
# 3. Loss_go：流动法则几何约束
# =========================
def loss_go(parts, phi):
    ux = parts['dx_dtheta']
    uy = parts['dy_dtheta']
    uz = parts['dz_dtheta']
    vx = parts['dx_dalpha']
    vy = parts['dy_dalpha']
    vz = parts['dz_dalpha']

    # n = ∂Γ/∂θ × ∂Γ/∂α
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    n_norm = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2 + 1e-9)

    theta = parts['pts'][:, 1:2]
    v_hat = torch.cat([
        torch.zeros_like(theta),
        -torch.cos(theta),
        -torch.sin(theta)
    ], dim=1)

    dot = (nx * v_hat[:, 0] + ny * v_hat[:, 1] + nz * v_hat[:, 2]) / n_norm[:, 0]

    return torch.mean((dot + math.sin(phi)) ** 2)


# =========================
# 4. Loss_bo：上/下界约束（Fig.5 里的平滑边界）
# =========================
def loss_bo(rc_up, rc_low, alpha, params, k=80.0):
    rd = params['rd']
    alpha_u = params['alpha_u']
    alpha_l = params['alpha_l']

    s_u = torch.tanh(k * (alpha - alpha_u))
    s_l = torch.tanh(k * (alpha - alpha_l))

    return torch.mean(((rc_up - rd) * (1 - s_u)) ** 2 +
                      ((rc_low - rd) * (1 + s_l)) ** 2)


# =========================
# 5. 训练（纯图 5 几何 PINN）
# =========================
def train_fig5(num_iters=4000, colloc=4000):
    params = {
        'r0': 1.0,
        'r0p': 0.8,
        'theta0': 0.0,
        'tanphi': math.tan(math.radians(25.0)),
        'rh': 0.5,
        'theta_h': 1.2,
        'alpha_u': 0.0,
        'alpha_l': math.pi,
        'rd': 0.2,
        'phi': math.radians(25.0)
    }

    net_up = RCNet().to(DEVICE)
    net_low = RCNet().to(DEVICE)

    optimizer = optim.Adam(
        list(net_up.parameters()) + list(net_low.parameters()),
        lr=1e-3
    )

    for it in range(1, num_iters + 1):
        # 随机采样 α–θ collocation 点
        alpha = torch.rand(colloc, 1, device=DEVICE) * 2 * math.pi
        theta = torch.rand(colloc, 1, device=DEVICE) * 1.2

        parts_up = compute_xyz_and_partials(net_up, alpha, theta, params)
        parts_low = compute_xyz_and_partials(net_low, alpha, theta, params)

        Lgo_up = loss_go(parts_up, params['phi'])
        Lgo_low = loss_go(parts_low, params['phi'])
        Lgo = 0.5 * (Lgo_up + Lgo_low)

        Lbo = loss_bo(parts_up['rc'], parts_low['rc'], alpha, params)

        loss = Lgo + 0.2 * Lbo

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == 1 or it % 200 == 0:
            print(f"Iter {it}/{num_iters}  Lgo={Lgo.item():.3e}  Lbo={Lbo.item():.3e}")

    return net_up, net_low, params


if __name__ == "__main__":
    net_u, net_l, params = train_fig5()
