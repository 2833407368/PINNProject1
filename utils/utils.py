import numpy as np
"""
Utility functions for the slope stability PINN workflow.
"""
import math
from typing import Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

import torch
from torchgen.api import autograd


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute ``dy/dx`` for scalar ``y`` with respect to ``x`` using autograd."""
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

def rtox(alpha, theta, rc, xm = 0, ym = 0, zm = 0):
    x = xm + rc * np.sin(theta)
    y = ym + rc * np.cos(alpha) * np.cos(theta)
    z = zm - rc * np.cos(alpha) * np.sin(theta)

def rtox_numpy(alpha, theta, r, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Convert spherical coordinates to Cartesian using NumPy arrays.

    NumPy is an optional dependency; if it is not available this function will
    raise a clear error so callers can fall back to torch equivalents.
    """
    if np is None:
        raise ImportError("NumPy is required for rtox_numpy but is not installed.")

    xm, ym, zm = center
    x = xm + r * np.sin(theta)
    y = ym + r * np.cos(alpha) * np.cos(theta)
    z = zm - r * np.cos(alpha) * np.sin(theta)
    return x, y, z


def rtox_torch(alpha: torch.Tensor, theta: torch.Tensor, r: torch.Tensor,
               center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert spherical coordinates to Cartesian coordinates using torch tensors."""
    xm, ym, zm = center
    x = xm + r * torch.sin(theta)
    y = ym + r * torch.cos(alpha) * torch.cos(theta)
    z = zm - r * torch.cos(alpha) * torch.sin(theta)
    return x, y, z


def surface_frame(alpha: torch.Tensor, theta: torch.Tensor, r: torch.Tensor,
                  center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Cartesian points and unit normals for the predicted slip surface."""
    x, y, z = rtox_torch(alpha, theta, r, center=center)
    point = torch.stack([x, y, z], dim=1)

    # Tangent vectors along alpha and theta directions
    t_alpha = torch.stack([
        gradients(x, alpha),
        gradients(y, alpha),
        gradients(z, alpha),
    ], dim=1)

    t_theta = torch.stack([
        gradients(x, theta),
        gradients(y, theta),
        gradients(z, theta),
    ], dim=1)

    normal = torch.cross(t_alpha, t_theta, dim=1)
    normal_norm = torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
    unit_normal = normal / normal_norm
    return point, unit_normal


def mohr_coulomb_residual(alpha: torch.Tensor, theta: torch.Tensor, r: torch.Tensor,
                          params: dict, center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> torch.Tensor:
    """Compute the Mohr–Coulomb yield residual on the predicted slip surface."""
    point, unit_normal = surface_frame(alpha, theta, r, center=center)
    depth = torch.clamp(-point[:, 2], min=0.0)

    vertical_alignment = unit_normal[:, 2].abs()
    sigma_n = params["gamma"] * depth * vertical_alignment

    shear_component = torch.sqrt(torch.clamp(1.0 - vertical_alignment ** 2, min=0.0))
    tau = params["gamma"] * depth * shear_component

    shear_capacity = params["c"] + sigma_n * math.tan(params["phi"])
    residual = tau - shear_capacity
    return residual