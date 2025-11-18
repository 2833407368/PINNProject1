import numpy as np
import torch
from torchgen.api import autograd


def gradients(y, x):
    return autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

def rtox(alpha, theta, rc, xm = 0, ym = 0, zm = 0):
    x = xm + rc * np.sin(theta)
    y = ym + rc * np.cos(alpha) * np.cos(theta)
    z = zm - rc * np.cos(alpha) * np.sin(theta)
    return x, y, z



