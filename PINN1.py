import torch
import torch.nn as nn
import torch.optim as optim
import math


# ==============================
# 1. 全连接 PINN 网络
# ==============================
class FCNet(nn.Module):
    """
    输入: (alpha, theta) 二维
    输出: rc  (标量)
    激活: tanh （论文也是用 tanh）
    """
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_hidden=6):
        super().__init__()
        layers = []
        last_dim = in_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, alpha, theta):
        # alpha, theta: [N, 1]
        x = torch.cat([alpha, theta], dim=-1)  # [N, 2]
        rc = self.net(x)                       # [N, 1]
        return rc


# ==============================
# 2. 边坡几何 + 3D 旋转机制参数
# ==============================
class SlopeGeom:
    """
    这里先实现均质土坡 + 旋转机制参数给定的情况。
    如果要完全复现论文，需要外层再套 PSO 来同时优化 (theta0, thetah, r0p_r0)。
    """

    def __init__(self,
                 H=5.0,         # 坡高
                 B_over_H=2.0,  # 宽高比 B/H
                 beta_deg=75.0, # 坡面倾角
                 phi_deg=15.0,  # 内摩擦角 φ
                 theta0_deg=20.0,
                 thetah_deg=100.0,
                 r0=1.0,
                 r0p_r0=0.2):
        self.H = H
        self.B_over_H = B_over_H
        self.beta = math.radians(beta_deg)
        self.phi = math.radians(phi_deg)

        self.theta0 = math.radians(theta0_deg)
        self.thetah = math.radians(thetah_deg)

        self.r0 = r0
        self.r0p = r0p_r0 * r0

        # 按式 (7)、(8) 可以求任意 theta 下的 r, r'
        # r(θ) = r0 * exp((θ-θ0)*tanφ)
        # r'(θ) = r0' * exp(-(θ-θ0)*tanφ)

        # 方便后面用：rh = r(θh)
        self.rh = self.r(self.thetah)

    def r(self, theta):
        return self.r0 * torch.exp((theta - self.theta0) * math.tan(self.phi))

    def rp(self, theta):
        return self.r0p * torch.exp(-(theta - self.theta0) * math.tan(self.phi))

    def midpoint_xyz(self, theta, r, rp):
        """
        对应附录公式 (A1)：
        xm = 0
        ym = 0.5(r+r')cosθ - rh cosθh
        zm = rh sinθh - 0.5(r+r')sinθ
        这里用 torch 写成向量形式。
        """
        xm = torch.zeros_like(theta)

        ym = 0.5 * (r + rp) * torch.cos(theta) - self.rh * math.cos(self.thetah)
        zm = self.rh * math.sin(self.thetah) - 0.5 * (r + rp) * torch.sin(theta)
        return xm, ym, zm

    def slope_surface_z(self, y):
        """
        简单实现一个平面坡面 z(y) 供筛选/约束用。
        假设坡脚在原点 (0,0,0)，坡面抬升 H，倾角 beta。
        y 轴沿着坡面水平方向。
        z = y * tan(beta)   (0 <= y <= H/tan(beta))
        实际论文几何可以根据你真实算例再细化。
        """
        return y * math.tan(self.beta)


# ==============================
# 3. 3D 坐标 & 法向量计算
# ==============================
def compute_xyz_rc(geom: SlopeGeom, net: FCNet, alpha, theta):
    """
    给定 alpha, theta，网络输出 rc，并根据式 (9) 计算 (x, y, z)。
    返回:
        rc: [N, 1]
        x, y, z: [N, 1]
    """
    rc = net(alpha, theta)  # [N, 1]
    r = geom.r(theta)       # [N, 1]
    rp = geom.rp(theta)     # [N, 1]

    xm, ym, zm = geom.midpoint_xyz(theta, r, rp)  # [N, 1] each

    # 式 (9)
    x = xm + rc * torch.sin(alpha)
    y = ym + rc * torch.cos(alpha) * torch.cos(theta)
    z = zm - rc * torch.cos(alpha) * torch.sin(theta)

    return rc, x, y, z, r, rp


def compute_normals_and_residual(geom: SlopeGeom, net: FCNet, alpha, theta):
    """
    用自动求导计算 dΓ/dα, dΓ/dθ，得到法向量 n，并算 PDE 残差:
        n̂ · v̂ + sinφ
    """
    alpha.requires_grad_(True)
    theta.requires_grad_(True)

    rc, x, y, z, r, rp = compute_xyz_rc(geom, net, alpha, theta)

    # 计算 dΓ/dθ
    grads_x_theta = torch.autograd.grad(x, theta, grad_outputs=torch.ones_like(x),
                                        retain_graph=True, create_graph=True)[0]
    grads_y_theta = torch.autograd.grad(y, theta, grad_outputs=torch.ones_like(y),
                                        retain_graph=True, create_graph=True)[0]
    grads_z_theta = torch.autograd.grad(z, theta, grad_outputs=torch.ones_like(z),
                                        retain_graph=True, create_graph=True)[0]

    # 计算 dΓ/dα
    grads_x_alpha = torch.autograd.grad(x, alpha, grad_outputs=torch.ones_like(x),
                                        retain_graph=True, create_graph=True)[0]
    grads_y_alpha = torch.autograd.grad(y, alpha, grad_outputs=torch.ones_like(y),
                                        retain_graph=True, create_graph=True)[0]
    grads_z_alpha = torch.autograd.grad(z, alpha, grad_outputs=torch.ones_like(z),
                                        retain_graph=True, create_graph=True)[0]

    # 拼成向量 [N, 3]
    dG_dtheta = torch.stack([grads_x_theta, grads_y_theta, grads_z_theta], dim=-1)
    dG_dalpha = torch.stack([grads_x_alpha, grads_y_alpha, grads_z_alpha], dim=-1)

    # n = dG/dθ × dG/dα
    n = torch.cross(dG_dtheta, dG_dalpha, dim=-1)  # [N, 3]
    n_norm = torch.norm(n, dim=-1, keepdim=True) + 1e-12
    n_hat = n / n_norm

    # v̂ = (0, -cosθ, -sinθ)
    v_hat = torch.stack([
        torch.zeros_like(theta.squeeze(-1)),
        -torch.cos(theta.squeeze(-1)),
        -torch.sin(theta.squeeze(-1))
    ], dim=-1)  # [N, 3]

    # n̂ · v̂
    dot_nv = (n_hat * v_hat).sum(dim=-1, keepdim=True)  # [N, 1]

    # 残差: n̂·v̂ + sinφ = 0
    residual = dot_nv + math.sin(geom.phi)  # [N, 1]
    return residual, rc, x, y, z, r, rp


# ==============================
# 4. Physics loss (PDE + 边界)
# ==============================
def pinn_rc_loss(geom: SlopeGeom, net: FCNet,
                 alpha, theta,
                 alpha_u=0.2*math.pi, alpha_l=1.8*math.pi):
    """
    alpha, theta: [N, 1]
    """
    residual, rc, x, y, z, r, rp = compute_normals_and_residual(geom, net, alpha, theta)

    # ---- (1) Governing Equation Loss (式 20) ----
    loss_go = (residual ** 2).mean()

    # ---- (2) Boundary Loss (简化版的式 21) ----
    # rd = (r - r') / 2
    rd = 0.5 * (r - rp)

    # 上边界: alpha < alpha_u 的点，让 rc → rd
    mask_up = (alpha < alpha_u).float()
    # 下边界: alpha > alpha_l 的点，让 rc → rd
    mask_low = (alpha > alpha_l).float()

    # 为了数值稳定，取平均时加一个 eps，避免分母为 0
    eps = 1e-8
    if mask_up.sum() > 0:
        loss_bo_up = (((rc - rd) * mask_up) ** 2).sum() / (mask_up.sum() + eps)
    else:
        loss_bo_up = torch.tensor(0.0, device=alpha.device)

    if mask_low.sum() > 0:
        loss_bo_low = (((rc - rd) * mask_low) ** 2).sum() / (mask_low.sum() + eps)
    else:
        loss_bo_low = torch.tensor(0.0, device=alpha.device)

    loss_bo = loss_bo_up + loss_bo_low

    loss = loss_go + loss_bo

    return loss, {'loss_go': loss_go.detach().item(),
                  'loss_bo': loss_bo.detach().item()}


# ==============================
# 5. 训练循环
# ==============================
def train_pinn_rc(
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_epochs=20000,
        batch_size=2048,
        lr=1e-3):

    geom = SlopeGeom(
        H=5.0,
        B_over_H=2.0,
        beta_deg=75.0,
        phi_deg=15.0,
        theta0_deg=23.9,   # 你可以用论文 case 13 的最优值
        thetah_deg=76.7,
        r0=1.0,
        r0p_r0=0.131
    )

    net = FCNet(in_dim=2, out_dim=1, hidden_dim=64, num_hidden=6).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        # 采样 alpha, theta
        # alpha ∈ [0, 2π], theta ∈ [theta0, thetah]
        alpha = (2 * math.pi) * torch.rand(batch_size, 1, device=device)
        theta = geom.theta0 + (geom.thetah - geom.theta0) * torch.rand(batch_size, 1, device=device)

        loss, loss_items = pinn_rc_loss(geom, net, alpha, theta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:6d} | "
                  f"Loss = {loss.item():.4e} | "
                  f"Loss_go = {loss_items['loss_go']:.4e} | "
                  f"Loss_bo = {loss_items['loss_bo']:.4e}")

    return net, geom


if __name__ == "__main__":
    net, geom = train_pinn_rc()
    # 训练完之后，你可以在 (alpha, theta) 网格上采样，导出 (x,y,z) 点云可视化滑动面。
