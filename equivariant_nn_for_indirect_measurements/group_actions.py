import torch
import numpy as np

from .groups import SE2, R2xGL
from .utils import R, Vec, angle_modulo, EPS

class DomainTransform():
    group = NotImplemented
    dim = NotImplemented
    origin = NotImplemented

    def apply(group_elems, points):
        raise NotImplementedError

    def lift(points, n=None):
        # return a group element that maps the origin to the input point
        raise NotImplementedError
    
    def project(group_elems):
        raise NotImplementedError
    
    def det(points=None, group_elems=None):
        # the functional determinant at the origin for the group elements,
        # as needed in main theorem conv.
        raise NotImplementedError
    
    def dist(points_a, points_b):
        # a metric on the space of the group action
        raise NotImplementedError


class GeneralizedDomainTransform(DomainTransform):
    def output_transform(group_elems, points):
        raise NotImplementedError


class SE2onR2(DomainTransform):
    """
    As the origin in R^2, we define (0,0).
    """
    group = SE2
    dim = 2
    origin = torch.zeros(2)

    def apply(group_elems, points):
        return group_elems[..., :2] + (R(group_elems[..., [2]]) @ points[..., None]).squeeze(-1)

    def lift(points, stabilizers=None):
        # this is not unique, we could choose other group elements that
        # project to the same point, choice can be defined by setting stabilizer

        # the stabilizers are assumed to be of shape (..., 1), containing only the angle
        angles = torch.zeros(*points.shape[:-1], 1, device=points.device) if stabilizers is None else stabilizers
        return torch.cat([points,
                          angles], dim=-1)
    
    def project(group_elems):
        # slower version that works for all groups:
        # return SE2onR2.apply(group_elems, SE2onR2.origin)
        # sped up version:
        return group_elems[..., :2]
    
    def det(points=None, group_elems=None):
        return 1.
    
    def rand_stabilizers(shape=[]):
        return
    
    def dist(points_a, points_b):
        return torch.norm(points_a-points_b, p=2, dim=-1, keepdim=True)
    
    def norm(a_inverse_b):
        return torch.norm(a_inverse_b, p=2, dim=-1, keepdim=True)

    def sample_uniform(radius, n_points, device='cpu'):
        random = torch.rand(n_points, 2, device=device)
        r = radius * torch.sqrt(random[..., 0])
        theta = 2*np.pi*random[..., 1]-np.pi
        return torch.stack([r*torch.cos(theta),
                            r*torch.sin(theta)], dim=-1)

class SE2onRadon(DomainTransform):
    """
    Group action of SE2 on the space of sinograms, R x [0, pi).
    We choose (0,0) as the origin.
    """
    group = SE2
    dim = 2
    origin = torch.zeros(2)

    def apply(group_elems, points):
        s = group_elems[..., :2]
        gamma = group_elems[..., [2]]
        r = points[..., [0]]
        phi = points[..., [1]]
        angle_sum = phi + gamma
        return torch.cat([r + torch.sum(s*Vec(angle_sum), dim=-1, keepdim=True),
                          angle_modulo(angle_sum)], dim=-1)
    
    def lift(points, stabilizers=None):
        # returns one or multiple group elements that maps (0,0) to the input points
        # assume stabilizers to be of shape (..., 1) and contain only lambdas (ex. 1.29)
        r = points[..., [0]]
        phi = points[..., [1]]
        if stabilizers is None:
            return torch.cat([r * Vec(phi),
                              phi], dim=-1)
        else:
            return torch.cat([r * Vec(phi) + stabilizers * Vec(phi + np.pi/2),
                              phi], dim=-1)
    
    def det(points=None, group_elems=None):
        return 1.
    
    def project(group_elems):
        # sped up version: (could be implemented even faster?)
        s = group_elems[..., :2]
        gamma = group_elems[..., [2]]
        return torch.cat([torch.sum(s*Vec(gamma), dim=-1, keepdim=True),
                          gamma], dim=-1)
    
    def dist(points_a, points_b):
        # there is no invariant metric on the Radon space
        raise NotImplementedError
    
    def norm(points):
        # we implement the group norm of lifted points
        phi = points[..., [1]]
        phi = angle_modulo(phi)#/(2*np.pi)
        r = points[..., [0]]
        return torch.clamp(torch.abs(phi)*torch.sqrt(2.+1./(2*(1.-torch.cos(phi))+EPS)*r**2),
                           min=r)

    def sample_uniform(radius, n_points, device='cpu'):
        shift = torch.tensor([radius, np.pi], device=device)
        scale = 2.*shift
        return torch.rand(n_points, 2, device=device) * scale - shift


class R2xGLonRadon(GeneralizedDomainTransform):
    group = R2xGL
    dim = 2
    origin = torch.zeros(2)

    def apply(group_elems, points):
        A_inv = torch.linalg.inv(R2xGL.A(group_elems[..., 2:]))
        vec_phi = Vec(points[..., [1]])
        A_mT_phi = (torch.transpose(A_inv, -2, -1) @ vec_phi.unsqueeze(-1)).squeeze(-1)
        s_A_phi = torch.sum(group_elems[..., :2] * A_mT_phi, dim=-1, keepdim=True)
        alpha = torch.linalg.vector_norm(A_mT_phi, dim=-1, keepdim=True)
        return torch.cat([(points[..., [0]] + s_A_phi)/alpha,
                          torch.arccos(A_mT_phi[..., [0]]/alpha)], dim=-1)

    def lift(points):
        # returns a group element that maps (0,0) to the input points
        r = points[..., [0]]
        phi = points[..., [1]]
        return torch.cat([r * Vec(phi),
                          phi,
                          torch.ones_like(r),
                          torch.zeros_like(r),
                          torch.ones_like(r)], dim=-1)

    def norm(points):
        # norm of lifted points
        r = points[..., [0]]
        phi = angle_modulo(points[..., [1]])
        return torch.sqrt(0.5*r**2 + 2*(1-torch.cos(phi)))
    
    def det(points=None, group_elems=None):
        """compute the functional determinant at the origin for certain group elements or
           the lifted versions of some points"""
        if points is not None:
            if group_elems is not None:
                raise ValueError
            # for lifted points, the determinant simplifies to
            return 1.
        else:
            if group_elems is None:
                raise ValueError

            A = torch.linalg.inv(R2xGL.A(group_elems[..., 2:]))
            a1 = A[..., 0, :]
            return torch.abs((torch.linalg.det(A)/torch.linalg.vector_norm(a1, dim=-1)**3).unsqueeze(-1))

    def output_transform(group_elems, points):
        det = group_elems[..., [3]] * group_elems[..., [5]] # a * d
        A_inv = torch.linalg.inv(R2xGL.A(group_elems[..., 2:]))
        vec_phi = Vec(points[..., [1]])
        A_mT_phi = (torch.transpose(A_inv, -2, -1) @ vec_phi.unsqueeze(-1)).squeeze(-1)
        alpha = torch.linalg.vector_norm(A_mT_phi, dim=-1, keepdim=True)
        return det / alpha

    def project(group_elems):
        return R2xGLonRadon.apply(group_elems,
                                  R2xGLonRadon.origin.to(group_elems.device))