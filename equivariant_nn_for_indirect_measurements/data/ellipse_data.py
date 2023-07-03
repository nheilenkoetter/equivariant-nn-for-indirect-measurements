import numpy as np
import torch
import astra

from ..utils import angle_modulo
from .data_preparation import FanBeamRadon

def radius(a, b, phi):
    return 1/(np.sqrt(np.cos(phi)**2/a**2+np.sin(phi)**2/b**2))

def min_max_thickness(a1, b1, a2, b2, phi_diff):
    phi = np.expand_dims(np.linspace(0, np.pi, int(np.pi*4000)), 0)
    radius1 = radius(np.expand_dims(a1, -1), np.expand_dims(b1, -1), phi)
    radius2 = radius(np.expand_dims(a2, -1), np.expand_dims(b2, -1), phi+np.expand_dims(phi_diff, -1))
    thickness = radius2 - radius1
    return np.min(thickness, axis=-1), np.max(thickness, axis=-1)

def c(theta, a, b, phi):
    return np.sqrt(a**2*np.cos(theta-phi)**2+b**2*np.sin(theta-phi)**2)

def t(r, theta, h, k):
    return r - h*np.cos(theta) + k*np.sin(theta)

def parallel_radon_ellipses(r, theta, a, b, phi, h, k):
    theta = np.expand_dims(theta, axis=-1)
    c_result = c(theta, a, b, phi)
    t_result = t(np.expand_dims(r, axis=-1), theta, h, k)
    return (2*a*b/c_result**2)*np.sqrt((c_result**2-t_result**2)*((np.abs(t_result) < c_result).astype(np.float32)))

def ellipse_image(x, y, a, b, phi, h, k):
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    return (((((x-h)*np.cos(phi)+(k-y)*np.sin(phi))/a)**2+((-(x-h)*np.sin(phi)+(k-y)*np.cos(phi))/b)**2) < 1).astype(np.float32)

def rotate90(image, axis, reversed=False):
    axis = list(axis)
    for i in (0,1):
        axis[i] = len(image.shape)+axis[i] if axis[i] < 0 else axis[i]
    axis = sorted(axis)
    if not reversed:
        return image.transpose(*axis).flip(axis[0])
    else:
        return image.flip(axis[0]).transpose(*axis)


def generate_ellipse_data(n_samples, seed, rotated, r, theta, im_shape=[129, 129]):
    np.random.seed(seed)
    torch.manual_seed(seed)
    a1_train = 0.5 + np.random.rand(n_samples)
    b1_train = 0.5 + np.random.rand(n_samples)
    phi1_train = 2*np.pi*np.random.rand(n_samples) + rotated
    a2_train = a1_train + 0.15 + 0.25 * np.random.rand(n_samples)
    b2_train = b1_train + 0.15 + 0.25 * np.random.rand(n_samples)
    phi_diff_train = 0.1*np.pi*(np.random.rand(n_samples)-0.5)
    phi2_train = phi1_train + phi_diff_train

    h_train_tmp = 0.3 * np.random.randn(n_samples)
    k_train_tmp = 0.3 * np.random.randn(n_samples)
    h_train = np.cos(rotated) * h_train_tmp + np.sin(rotated) * k_train_tmp
    k_train = -np.sin(rotated) * h_train_tmp + np.cos(rotated) * k_train_tmp

    min_train, max_train = min_max_thickness(a1_train, b1_train, a2_train, b2_train, phi_diff_train)
    label = torch.stack([torch.Tensor(min_train), torch.Tensor(max_train)], dim=-1)
    #label = np.stack([a1_train, b1_train, a2_train, b2_train, phi_diff_train], axis=-1)

    ellipse1_train = parallel_radon_ellipses(r, theta, a1_train, b1_train, phi1_train, h_train, k_train)
    ellipse2_train = parallel_radon_ellipses(r, theta, a2_train, b2_train, phi2_train, h_train, k_train)
    measurement_train = torch.Tensor(ellipse2_train-ellipse1_train).moveaxis(-1,0).unsqueeze(-1)

    x, y = np.meshgrid(np.linspace(-3, 3, im_shape[0]), np.linspace(-3, 3, im_shape[1]))
    ellipse1_image = ellipse_image(x, y, a1_train, b1_train, phi1_train, h_train, k_train)
    ellipse2_image = ellipse_image(x, y, a2_train, b2_train, phi2_train, h_train, k_train)
    images = ellipse2_image-ellipse1_image

    sinogram = measurement_train / torch.max(measurement_train)
    sinogram = sinogram + 0.05*torch.randn_like(sinogram)
    
    r, theta = torch.Tensor(r), torch.Tensor(theta)
    # correctly add the full information (use periodicity)
    points = torch.stack([r, theta], dim=-1)
    periodic_points = torch.stack([-r, angle_modulo(theta-np.pi)], dim=-1)
    points = torch.cat([points, periodic_points], dim=-3).unsqueeze(0).reshape(1, -1, 2) # TODO: check meaning of dims
    values = torch.cat([sinogram, sinogram], dim=-3).reshape(sinogram.shape[0], -1, 1)
    return points, values, label, images, sinogram


class EllipseDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=60000, seed=42, rotated=0):
        self.n_samples = n_samples
        self.seed = seed
        self.rotated = rotated

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.values[idx], self.minmax[idx]

class EllipseParallelRadonDataset(EllipseDataset):
    def __init__(self, n_samples=60000, seed=42, rotated=0, n_r=64, n_theta=2):
        # uniform grid in [-2.5, 2.5] x [0, pi) with n_x and n_y points
        super().__init__(n_samples, seed, rotated)
        r, theta = np.meshgrid(np.linspace(-2.5, 2.5, n_r), np.linspace(0, np.pi, n_theta, endpoint=False))
        self.points, self.values, self.minmax, self.images, _ = generate_ellipse_data(self.n_samples, self.seed, self.rotated, r, theta)

class EllipseFanRadonDataset(EllipseDataset):
    def __init__(self, alpha, beta, D, n_samples=60000, seed=42, rotated=0, im_shape=[129, 129]):
        super().__init__(n_samples, seed, rotated)
        alpha, beta = np.meshgrid(alpha, beta)
        r = D*np.sin(alpha)
        theta = angle_modulo(beta-alpha)
        self.points, values, self.minmax, self.images, self.sino_images = generate_ellipse_data(self.n_samples, self.seed, self.rotated, r, theta, im_shape=im_shape)
        self.values = (values - torch.mean(values)) / torch.mean(torch.std(values, dim=-2))

class RecoDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, im_shape, im_size, D_source, D_det, dist_det, N_det, angles):
        self.images = images
        self.labels = labels

        self.radon = FanBeamRadon(im_shape=im_shape,
                                  im_size=im_size,
                                  D_source=D_source,
                                  D_det=D_det,
                                  dist_det=dist_det,
                                  N_det=N_det,
                                  angles=angles)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        _, reco = astra.create_reconstruction('SIRT', self.radon.projector_id, np.array(self.images[idx,...,0]), iterations=5)
        reco = (reco - 0.03)/0.035
        return reco[..., None], self.labels[idx]

def ellipse_fan_radon_astra(im_shape, im_size, D_source, D_det, dist_det, N_det, angles, n_samples=10000, seed=42):
    det_space = np.linspace(-dist_det*(N_det-1)/2, dist_det*(N_det-1)/2, N_det, endpoint=True)
    alpha = np.arctan(det_space/(D_source+D_det))
    points_ds = EllipseFanRadonDataset(alpha, angles, D_source, n_samples, seed, rotated=0, im_shape=im_shape)
    reco_ds = RecoDataset(points_ds.sino_images,
                          points_ds.minmax,
                          im_shape,
                          im_size,
                          D_source,
                          D_det,
                          dist_det,
                          N_det,
                          angles)
    return points_ds, reco_ds