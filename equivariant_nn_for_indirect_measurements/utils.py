import torch
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-6

def R(gamma):
    # assumes shape (..., 1) for gamma
    sinus = torch.sin(gamma)
    cosinus = torch.cos(gamma)
    return torch.stack([torch.cat([cosinus, -sinus], dim=-1),
                        torch.cat([sinus, cosinus], dim=-1)], dim=-2)

def Vec(gamma):
    # assumes shape (..., 1) for gamma
    return torch.cat([torch.cos(gamma), torch.sin(gamma)], dim=-1)

def angle_modulo(gamma):
    # we add an EPS to force all values very close to pi to be set to -pi
    if isinstance(gamma, torch.Tensor):
        return torch.remainder(gamma+np.pi+EPS, 2*np.pi)-np.pi
    elif isinstance(gamma, np.ndarray):
        return np.remainder(gamma+np.pi+EPS, 2*np.pi)-np.pi
    else:
        raise ValueError

def scatter(points, values):
    fig = plt.figure()
    if points.shape[-1] == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points[..., 0], points[..., 1], points[..., 2], c=values)
        return fig, ax
    elif points.shape[-1] == 2:
        ax = fig.add_subplot()
        ax.scatter(points[..., 0], points[..., 1], c=values)
        return fig, ax
    else:
        raise NotImplementedError

def qr_decomposition(A):
    a1 = A[..., 0]
    a2 = A[..., 1]
    norm_a1 = torch.norm(a1, dim=-1, keepdim=True)
    u12 = torch.sum(a1 * a2, dim=-1, keepdim=True)/norm_a1
    u2 = a2 - u12/norm_a1 * a1
    e1 = a1/norm_a1
    e2 = u2/torch.norm(u2, dim=-1, keepdim=True)
    Q = torch.stack([e1, e2], dim=-1)
    U = torch.stack([torch.cat([norm_a1, torch.zeros_like(norm_a1)], dim=-1),
                     torch.cat([u12, torch.sum(e2 * a2, dim=-1, keepdim=True)], dim=-1)], dim=-1)
    return Q, U