import torch
from .networks.utils import BatchNorm
from .group_actions import GeneralizedDomainTransform
from .groups import Group
import matplotlib.pyplot as plt
import time

class KernelBasis(torch.nn.Module):
    def __init__(self, group_action_in, group_action_out, num_elems):
        # group action is the group action in the output
        # num_elements is the size of this basis
        super().__init__()
        self.group_action_in = group_action_in
        self.group_action_out = group_action_out
        self.num_elems = num_elems

class LocalFCNBasis(KernelBasis):
    def __init__(self, num_elems, width=32, crop_size=1.):
        super().__init__(None, None, num_elems)
        self.crop_size = crop_size
        self.width = width
    
    def setup(self, representation_in, representation_out):
        self.representation_in = representation_in
        self.representation_out = representation_out
        assert issubclass(self.representation_out, Group) # other cases not implemented yet
        self.group = self.representation_out

        self.net = torch.nn.Sequential(
            BatchNorm(self.representation_in.dim),
            torch.nn.Linear(self.representation_in.dim, self.width),
            torch.nn.SiLU(),
            BatchNorm(self.width),
            torch.nn.Linear(self.width, self.num_elems),
            torch.nn.SiLU()
        )
        return self

    def forward(self, g):
        if not issubclass(self.representation_in, Group):
            inverse_g = self.group.inverse(g)
            projected_g = self.representation_in.project(inverse_g)
            a = self.net(projected_g)
            if issubclass(self.representation_in, GeneralizedDomainTransform):
                a = a / self.representation_in.output_transform(g, self.representation_in.origin.to(g.device))
            det = self.representation_in.det(group_elems=inverse_g)
            if isinstance(det, torch.Tensor):
                a = a * det
            return a * torch.exp(-self.representation_in.norm(projected_g)**2/self.crop_size**2)
        else:
            a = self.net(g)
            return a * torch.exp(-self.representation_in.norm(g)**2/self.crop_size**2)


class Conv(torch.nn.Module):
    def __init__(self, representation_in, representation_out, channels_in, channels_out, kernel_basis, bias=True, knn_points=-1):
        super().__init__()

        self.knn_points = knn_points
        self.knn_mode = self.knn_points > 0

        assert issubclass(representation_out, Group)
        self.group = representation_out

        self.representation_in = representation_in
        self.representation_out = representation_out
        self.kernel_basis = kernel_basis.setup(self.representation_in, self.representation_out)
        self.ch_in = channels_in
        self.ch_out = channels_out
        self.coeff = torch.nn.Parameter(torch.randn(self.ch_in,
                                                    self.ch_out,
                                                    self.kernel_basis.num_elems))
        self.use_bias = bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.randn(self.ch_out))

    def forward(self, points_in, values_in, points_out, existing_u_inverse_v=None):
        # points_in (..., n_points_in, self.representation_in.dim), values_in (..., n_points_in, channels_in), points_out (..., n_points_out, self.representation_out.dim)
        if existing_u_inverse_v is None:
            with torch.no_grad():
                u_lifted = self.representation_in.lift(points_in[..., None, :, :])
                inverse = self.group.inverse(u_lifted)
                u_inverse_v = self.representation_out.apply(inverse, points_out[..., :, None, :]) # (..., n_points_out, n_points_in, self.representation_out.dim)
                if self.knn_mode:
                    norms = self.representation_out.norm(u_inverse_v)
                    _, indices = torch.topk(norms, k=self.knn_points, dim=-2, largest=False)
                    u_inverse_v = torch.gather(u_inverse_v, -2, indices.expand(*((len(indices.shape)-1)*[-1]), self.representation_out.dim)) # (..., n_points_out, knn, self.representation_out.dim)
        else:
            if self.knn_mode:
                u_inverse_v, indices = existing_u_inverse_v
            else:
                u_inverse_v = existing_u_inverse_v

        det = self.representation_in.det(points=points_in[..., None, :, :])
        if isinstance(det, torch.Tensor):
            values_in = values_in / det

        if issubclass(self.representation_in, GeneralizedDomainTransform):
            if existing_u_inverse_v is not None:
                u_lifted = self.representation_in.lift(points_in[..., None, :, :])
                inverse = self.group.inverse(u_lifted)
            values_in = self.representation_in.output_transform(inverse, self.representation_in.origin.to(values_in.device)) * values_in
        values_in = values_in.unsqueeze(-3) # (..., 1, n_points_in, channels_in)

        if self.knn_mode:
            values_in = values_in.expand(*(len(values_in.shape)-3)*[-1], u_inverse_v.shape[-3], -1, -1) # (..., n_points_out, n_points_in, channels_in)
            new_shape = list(values_in.shape)
            new_shape[-2] = indices.shape[-2]
            values_in = torch.gather(values_in, -2, indices.unsqueeze(0).expand(*new_shape)) # (..., n_points_out, knn, channels_in)
        kernel_b = self.kernel_basis(u_inverse_v)

        # optimize (using opt_einsum as pytorch backend) the 3 matrix-multiplications automatically
        # this will be equivalent to the pointconv-trick
        torch.backends.opt_einsum.strategy = 'optimal' # only for new pytorch versions
        # i channels_in, o channels_out, e basis.num_elems, u n_points_in, v n_points_out
        out = torch.einsum('ioe, ...vue, ...vui -> ...vo', self.coeff, kernel_b, values_in) / values_in.shape[-2] # average instead of summation

        if self.knn_mode:
            return (u_inverse_v, indices), (out + self.bias if self.use_bias else out)
        else:
            return (u_inverse_v), (out + self.bias if self.use_bias else out)