import astra
import numpy as np
from PIL import Image
import torch

from ..utils import angle_modulo

class ZeroPad():
    def __call__(self, image):
        assert image.size[0] == image.size[1]
        size = image.size[0]
        if size % 2 != 1:
            result = Image.new(image.mode, (size+1, size+1), 0)
            result.paste(image, (0, 0))
            return result
        return image

class FanBeamRadon():
    def __init__(self, im_shape, im_size=[2., 2.], D_source=2.5, D_det=2.5, dist_det=0.1, N_det=29, angles=[0., np.pi/2], range=1., noise_level=0., mean=0., std=1.):
        self.range = range
        self.noise_level = noise_level
        assert (im_shape[0] % 2 == 1) and (im_shape[1] % 2 == 1)
        self.vol_geom = astra.create_vol_geom(im_shape[0], im_shape[1], -im_size[0]/2, im_size[0]/2, -im_size[1]/2, im_size[1]/2)
        self.proj_geom = astra.create_proj_geom('fanflat', dist_det, N_det, angles, D_source, D_det)
        self.projector_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)

        self.mean = mean
        self.std = std

        grid_det, beta = np.meshgrid(np.linspace(-dist_det*(N_det-1)/2, dist_det*(N_det-1)/2, N_det, endpoint=True),
                                     angles)
        alpha = np.arctan(grid_det/(D_source+D_det))
        r = D_source*np.sin(alpha)
        theta = angle_modulo(beta-alpha)
        points = np.stack([r, theta], axis=-1)
        points = points.reshape(-1, 2)
        periodic_points = np.stack([-r, angle_modulo(theta-np.pi)], axis=-1).reshape(-1, 2)
        self.points = torch.tensor(np.concatenate([points, periodic_points], axis=-2).astype(np.float32))

    def __call__(self, image):
        _, sinogram = astra.create_sino(np.array(image), self.projector_id)
        sinogram = sinogram + self.noise_level*self.range*np.random.randn(*sinogram.shape)
        values = sinogram.reshape(-1, 1)
        values = np.concatenate([values, values], axis=-2)#/np.max(values)
        return (torch.tensor(values.astype(np.float32))-self.mean)/self.std


class ParallelBeamRadon():
    def __init__(self, im_shape, im_size=[2., 2.], dist_det=0.1, N_det=29, angles=[0., np.pi/2], range=1., noise_level=0., mean=0., std=1.):
        self.range = range
        self.noise_level = noise_level
        assert (im_shape[0] % 2 == 1) and (im_shape[1] % 2 == 1)
        self.vol_geom = astra.create_vol_geom(im_shape[0], im_shape[1], -im_size[0]/2, im_size[0]/2, -im_size[1]/2, im_size[1]/2)
        self.proj_geom = astra.create_proj_geom('parallel', dist_det, N_det, angles)
        self.projector_id = astra.create_projector('line', self.proj_geom, self.vol_geom)

        self.mean = mean
        self.std = std


        r, theta = np.meshgrid(np.linspace(-dist_det*(N_det-1)/2, dist_det*(N_det-1)/2, N_det, endpoint=True),
                               angles)
        points = np.stack([r, angle_modulo(theta)], axis=-1)
        points = points.reshape(-1, 2)
        periodic_points = np.stack([-r, angle_modulo(theta-np.pi)], axis=-1).reshape(-1, 2)
        self.points = torch.tensor(np.concatenate([points, periodic_points], axis=-2).astype(np.float32))

    def __call__(self, image):
        _, sinogram = astra.create_sino(np.array(image), self.projector_id)
        sinogram = sinogram + self.noise_level*self.range*np.random.randn(*sinogram.shape)
        values = sinogram.reshape(-1, 1)
        values = np.concatenate([values, values], axis=-2)
        return (torch.tensor(values.astype(np.float32))-self.mean)/self.std

class RadonReconstruction():
    def __init__(self, radon, mean, std):
        self.radon = radon
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        _, sinogram = astra.create_sino(np.array(image), self.radon.projector_id)
        sinogram = sinogram + self.radon.noise_level*self.radon.range*np.random.randn(*sinogram.shape)
        _, reco = astra.create_reconstruction('SIRT', self.radon.projector_id,
                                              sinogram,
                                              iterations=5)
        return (reco[None]-self.mean)/self.std