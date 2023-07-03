import torch
import numpy as np
import matplotlib.pyplot as plt

from equivariant_nn_for_indirect_measurements.groups import SE2
from equivariant_nn_for_indirect_measurements.conv import Conv, LocalFCNBasis
from equivariant_nn_for_indirect_measurements.networks.utils import AvgPool, BatchNorm


class InvariantNet(torch.nn.Module):
    def __init__(self, representation_in, radius, input_points, out_classes, knn, initial_channels, initial_points, initial_kernel_radius, downsampling_factor, n_blocks=2,
                 inner_representation=SE2):
        super().__init__()
        self.representation_in = representation_in
        self.out_classes = out_classes
        self.radius = radius
        self.n_blocks = n_blocks
        self.inner_representation = inner_representation

        self.knn = knn
        self.n_points = [int(initial_points*(1/downsampling_factor)**i) for i in range(self.n_blocks)]
        self.radii = [initial_kernel_radius*downsampling_factor**(i/2) for i in range(self.n_blocks)]
        self.n_channels = [int(initial_channels*downsampling_factor**i) for i in range(self.n_blocks)]

        self.n_basis_elem = 16

        self.register_buffer('input_points', input_points)

        self.blocks = torch.nn.ModuleList(
            [torch.nn.ModuleList(
                [Conv(self.inner_representation, self.inner_representation, self.n_channels[i], self.n_channels[i], LocalFCNBasis(self.n_basis_elem,
                                                                                                crop_size=self.radii[i]),
                                                                                                knn_points=self.knn),
                 Conv(self.inner_representation, self.inner_representation, self.n_channels[i], self.n_channels[i], LocalFCNBasis(self.n_basis_elem,
                                                                                                crop_size=self.radii[i]),
                                                                                                knn_points=self.knn),
                 Conv(self.inner_representation, self.inner_representation, self.n_channels[i], self.n_channels[i], LocalFCNBasis(self.n_basis_elem,
                                                                                                crop_size=self.radii[i]),
                                                                                                knn_points=self.knn)
                ]
            ) for i in range(self.n_blocks)]
        )

        self.downsampling_layers = torch.nn.ModuleList([
            Conv(self.inner_representation, self.inner_representation, self.n_channels[i], self.n_channels[i+1], LocalFCNBasis(self.n_basis_elem,
                                                                                             crop_size=self.radii[i+1]),
                                                                                             knn_points=self.knn)
            for i in range(self.n_blocks-1)
        ])

        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.ModuleList([BatchNorm(self.n_channels[i]),
                                  BatchNorm(self.n_channels[i]),
                                  BatchNorm(self.n_channels[i]),
                                  BatchNorm(self.n_channels[i])]) for i in range(self.n_blocks)])

        self.first_layer = Conv(self.representation_in, self.inner_representation, 1, self.n_channels[0], LocalFCNBasis(self.n_basis_elem,
                                                                                                       crop_size=self.radii[0]),
                                                                                                       knn_points=self.knn) # important for computation along orbits (inactive)
        self.last_batch_norm = BatchNorm(self.n_channels[-1])
        self.last_layer = Conv(self.inner_representation, self.inner_representation, self.n_channels[-1], out_classes, LocalFCNBasis(self.n_basis_elem,
                                                                                                   crop_size=self.radii[-1]),
                                                                                                   knn_points=self.knn)
    
        self.pool = AvgPool()
    
    def group_points(self, radius, n, device):
        return self.inner_representation.sample_uniform(radius, n, device)

    def forward(self, values):
        device = self.input_points.device
        points =  self.group_points(self.radius, self.n_points[0], device)
        _, values = self.first_layer(self.input_points, values, points)
        values = torch.nn.functional.relu(values)

        for i in range(self.n_blocks):
            values_in = self.batch_norms[i][0](values)
            u_inverse_v, values = self.blocks[i][0](points, values_in, points)
            values = torch.nn.functional.relu(values)
            for j in range(1, len(self.blocks[i])):
                values = self.batch_norms[i][j](values)
                _, values = self.blocks[i][j](points, values, None, u_inverse_v)
                values = torch.nn.functional.relu(values)
            # residual connection
            values = values + values_in

            if i < self.n_blocks-1:
                next_points = self.group_points(self.radius, self.n_points[i+1], device)
                values = self.batch_norms[i][-1](values)
                _, values = self.downsampling_layers[i](points, values, next_points)
                values = torch.nn.functional.relu(values)

                points = next_points
        
        values = self.last_batch_norm(values)
        _, values = self.last_layer(points, values, points)

        return self.pool(values)