# Equivariant Neural Networks for Indirect Measurements

PyTorch library for the construction of Neural Networks that are equivariant with respect to induced symmetry representations in the measurement space of inverse problems.

This repository provides the code accompanying the preprint [Equivariant Neural Networks for Indirect Measurements](https://arxiv.org/abs/2306.16506) by Matthias Beckmann and Nick Heilenkötter, 2023.

![The action of induced group representations on the sinogram of a digit.](https://github.com/nheilenkoetter/equivariant-nn-for-indirect-measurements/blob/main/induced_representations.gif)

In inverse problems such as computed tomography, well-known symmetry actions (e.g. rotation, scaling, translation) are translated into new transforms inside the measurements (sinograms). The incorporation of this information into a neural network allows to build efficient architectures that operate directly on indirect measurements.