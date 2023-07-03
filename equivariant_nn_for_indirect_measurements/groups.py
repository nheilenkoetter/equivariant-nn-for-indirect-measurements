import torch
import numpy as np

from .utils import R, angle_modulo, qr_decomposition, EPS

class Group():
    dim = NotImplemented
    one = NotImplemented
    group = NotImplemented

    def mult(group_elems_a, group_elems_b):
        # assumes shape (..., dim) for group elements
        raise NotImplementedError
    
    def inverse(group_elems):
        raise NotImplementedError

class SE2(Group):
    """
    Group of rotations and translations in R^2.
    Assumes shape (..., 3) for group elements, where the first 2 entries
    in the last dimension correspond to a translation and the last entry
    is the angle of rotation in [0, 2*pi).
    """
    dim = 3
    one = torch.zeros(dim)

    def mult(group_elems_a, group_elems_b):
        s_a = group_elems_a[..., :2]
        s_b = group_elems_b[..., :2]
        gamma_a = group_elems_a[..., [2]]
        gamma_b = group_elems_b[..., [2]]
        return torch.cat([s_a + (R(gamma_a) @ s_b[..., None]).squeeze(-1),
                          angle_modulo(gamma_a+gamma_b)], dim=-1)
    
    def apply(group_elems, points):
        return SE2.mult(group_elems, points)
    
    def inverse(group_elems):
        s = group_elems[..., :2]
        gamma = group_elems[..., [2]]
        return torch.cat([-(R(-gamma) @ s[..., None]).squeeze(-1),
                          angle_modulo(-gamma)], dim=-1)

    def norm(a_inverse_b):
        angles = a_inverse_b[..., [2]]
        angles = angle_modulo(angles)#/(2*np.pi)
        sq_norm = torch.sum(a_inverse_b[..., :2]**2, dim=-1, keepdim=True)
        return torch.clamp(torch.abs(angles)*torch.sqrt(2.+1./(2*(1.-torch.cos(angles))+EPS)*sq_norm),
                           min=torch.sqrt(sq_norm))

    def det(group_elems=None, points=None):
        return 1.

    def sample_uniform(radius, n_points, device='cpu'):
        random = torch.rand(n_points, 3, device=device)
        r = radius * torch.sqrt(random[..., 0])
        theta = 2*np.pi*random[..., 1]
        return torch.stack([r*torch.cos(theta),
                            r*torch.sin(theta),
                            2*np.pi*random[..., 2]-np.pi], dim=-1)

    def lift(points):
        return points
    
    def project(group_elems):
        return group_elems

class R2xGL(Group):
    """
    Group of rotations and translations and anisotropic scaling in R^2.
    Assumes shape (..., 6) for group elements, where:
    the first to elements correspond to a translation
    R(phi) * (a b)
             (  d) 
    (x y phi a b d)
    """
    dim = 6
    one = torch.tensor([0., 0., 0., 1., 0., 1.])

    def A(params):
        # assumes shape (..., 1) for gamma
        phi = params[..., [0]]
        phi, a, b, d = [params[..., [i]] for i in range(4)]
        z = torch.zeros_like(a)
        abd = torch.stack([torch.cat([a, b], dim=-1),
                           torch.cat([z, d], dim=-1)], dim=-2)
        out = R(phi) @ abd
        return out
    
    def params(A):
        Q, U = qr_decomposition(A)
        diag_sign = torch.sign(U[..., [0, 1], [0, 1]])
        Q = Q * diag_sign[..., None, :]
        U = U * diag_sign[..., :, None]
        phi = torch.atan2(Q[..., [1], 0], Q[..., [0], 0])
        return torch.cat([phi,
                          U[..., [0], 0],
                          U[..., [0], 1],
                          U[..., [1], 1]], dim=-1)

    def mult(group_elems_a, group_elems_b):
        s_a = group_elems_a[..., :2]
        s_b = group_elems_b[..., :2]
        A_a = R2xGL.A(group_elems_a[..., 2:])
        A_b = R2xGL.A(group_elems_b[..., 2:])
        A_product = A_a @ A_b
        out_params = R2xGL.params(A_product)
        return torch.cat([s_a + (A_a @ s_b[..., None]).squeeze(-1),
                          out_params], dim=-1)

    def apply(group_elems, points):
        return R2xGL.mult(group_elems, points)
    
    def lift(points):
        return points
    
    def project(group_elems):
        return group_elems
    
    def inverse(group_elems):
        s = group_elems[..., :2]
        A = R2xGL.A(group_elems[..., 2:])
        A_inv = torch.linalg.inv(A)
        params_inv = R2xGL.params(A_inv)
        return torch.cat([(-A_inv @ s[..., None]).squeeze(-1),
                          params_inv], dim=-1)

    def det(points=None, group_elems=None):
        if points is not None:
            assert group_elems is None
            group_elems = points
        A_a = R2xGL.A(group_elems[..., 2:])
        origin = torch.tensor([[0., 1., 0., 1.]], device=group_elems.device).repeat(*group_elems.shape[:2], 1) # phi, a, b, d
        with torch.enable_grad():
            origin.requires_grad = True
            A_b = R2xGL.A(origin)
            result = A_a @ A_b
            out = torch.sum(R2xGL.params(result), dim=(0, 1))
        
            grads = []
            for i in range(4):
                grads.append(torch.autograd.grad(out[i], origin, retain_graph=i<3)[0])
        jac = torch.stack(grads, dim=-1).detach()

        return torch.abs(group_elems[..., [3]] * group_elems[..., [5]] * torch.linalg.det(jac).unsqueeze(-1))
    
    def sample_uniform(radius, n_points, device='cpu'):
        random = torch.rand(n_points, 6, device=device)
        r = radius * torch.sqrt(random[..., 0])
        theta = 2*np.pi*random[..., 1]
        return torch.stack([r*torch.cos(theta),
                            r*torch.sin(theta),
                            2*np.pi*random[..., 2]-np.pi,
                            0.75+0.5*random[..., 3],
                            -0.5+random[..., 4],
                            0.75+0.5*random[..., 5]], dim=-1)

    def _sq_spectral_norm(A):
        # computes the squared spectral norm of 2x2 matrices
        sq_fro = torch.sum(A**2, dim=(-2, -1))
        det = torch.linalg.det(A)
        return ((sq_fro+torch.sqrt(sq_fro**2-4*det**2))/2.).unsqueeze(-1)

    def norm(a_inverse_b):
        sq_translation_norm = torch.sum(a_inverse_b[..., :2]**2, dim=-1, keepdim=True)
        A = R2xGL.A(a_inverse_b[..., 2:])
        sq_spectral_norm = R2xGL._sq_spectral_norm(torch.eye(2, device=a_inverse_b.device)-A)
        return torch.sqrt(0.5*sq_translation_norm + sq_spectral_norm) # weights chosen randomly