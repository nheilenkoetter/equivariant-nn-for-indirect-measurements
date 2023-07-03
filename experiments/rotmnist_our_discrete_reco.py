# %%
from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt

from equivariant_nn_for_indirect_measurements.groups import SE2
from equivariant_nn_for_indirect_measurements.group_actions import SE2onR2, SE2onRadon, SE2onSE2
from equivariant_nn_for_indirect_measurements.conv import Conv, LocalFCNBasis
from equivariant_nn_for_indirect_measurements.networks.utils import AvgPool, BatchNorm
from equivariant_nn_for_indirect_measurements.data.data_preparation import ParallelBeamRadon, FanBeamRadon, RadonReconstruction
from equivariant_nn_for_indirect_measurements.data.rot_mnist import MnistRotDataset
from equivariant_nn_for_indirect_measurements.networks.resnet import SE2InvariantNet

import os
from torchvision.datasets import MNIST
from torchvision import transforms
PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select GPUs to use

r = 2.
batch_size = 14
n_angles = 9


import torchvision.transforms.functional as TF
import random
from PIL import Image

class ZeroPad:
    def __call__(self, image):
        result = Image.new(image.mode, (29, 29), 0)
        result.paste(image, (0, 0))
        return result

#radon = ParallelBeamRadon(im_shape=[29, 29], im_size=[2*r, 2*r], dist_det=r/14, N_det=29, angles=np.array([0., np.pi/2]))
max_angle = 2*np.pi if n_angles != 2 else np.pi
radon = FanBeamRadon(im_shape=[29, 29], im_size=[2*r, 2*r], D_det=3., D_source=3., dist_det=r/10, N_det=29, angles=np.linspace(0,max_angle, n_angles, endpoint=False), range=3.5558, noise_level=0.06, std=0.4328, mean=0.6933)
reco = RadonReconstruction(radon, mean=0.1568, std=0.15)


ds = MnistRotDataset(root='/home/nick7/equivariance/equivariant-nn-for-indirect-measurements/experiments/data/', train=True, transform=transforms.Compose([ZeroPad(), reco]))
#val_ds = MnistRotDataset(root='/home/nick7/equivariance/equivariant-nn-for-indirect-measurements/experiments/data/', train=False, transform=transforms.Compose([ZeroPad(), radon]))
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = torch.utils.data.random_split(ds, [10000, 2000], generator=generator)
# %%
#net = SE2InvariantNet(SE2onRadon, 10).to(torch.float32)


def rotate90(image, axis, reversed=False):
    axis = list(axis)
    for i in (0,1):
        axis[i] = len(image.shape)+axis[i] if axis[i] < 0 else axis[i]
    axis = sorted(axis)
    if not reversed:
        return image.transpose(*axis).flip(axis[0])
    else:
        return image.flip(axis[0]).transpose(*axis)

class ImageLifting(torch.nn.Module):
    def forward(self, x):
        #x = x.moveaxis(-1, -3)
        out = torch.stack([x, x, x, x], dim=-1)
        return out

class TwoAngleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = spatial_stride
        self.param = torch.nn.Parameter(torch.randn((self.out_channels, self.in_channels,3,3,3)))
        self.bias = torch.nn.Parameter(torch.randn((self.out_channels,)))

    def forward(self, x):
        # pad
        padded = torch.zeros((*x.shape[:-1], 6), device=x.device)
        padded[..., 0] = x[..., -1]
        padded[..., 1:-1] = x
        padded[..., -1] = x[..., 0]
        # prepare conv filters
        # new idea: avoid iteration by using a single axis for angles and input and
        # output channels, i.e. and sorting them again afterwards
        weight = torch.zeros((self.out_channels, self.in_channels, 4, 6, 3, 3), device=x.device)
        weight[:, :, 0, 0:3, :, :] = self.param
        weight[:, :, 1, 1:4, :, :] = rotate90(weight[:, :, 0, 0:3, :, :], (-1, -2))
        weight[:, :, 2, 2:5, :, :] = rotate90(weight[:, :, 1, 1:4, :, :], (-1, -2))
        weight[:, :, 3, 3:6, :, :] = rotate90(weight[:, :, 2, 2:5, :, :], (-1, -2))
        bias_arg = self.bias.repeat_interleave(4)
        # merge axis
        input = padded.moveaxis(-1, -3)
        input = input.reshape(*input.shape[:-4], self.in_channels*6, *input.shape[-2:])
        weight = weight.moveaxis(2, 1)
        weight = weight.reshape(self.out_channels*4, self.in_channels*6, *weight.shape[4:])
        # conv
        out = torch.nn.functional.conv2d(input, weight, bias=bias_arg, stride=self.stride, padding=1)
        return out.reshape(*x.shape[:-4], self.out_channels, 4, out.shape[-2], out.shape[-1]).moveaxis(-3, -1)

class TwoAngleResidualBlock(torch.nn.Module):
    def __init__(self, n_channels, n_hidden_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm3d(n_hidden_channels),
            TwoAngleConv(n_channels, n_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(n_hidden_channels),
            TwoAngleConv(n_hidden_channels, n_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(n_hidden_channels),
            TwoAngleConv(n_hidden_channels, n_channels),
            torch.nn.ReLU()
        )
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.layers(x))    #[..., 3:-3, 3:-3, :]


class GlobalAvgPool3d(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(2,3,4))

n_ch = 22

save_path = 'rotmnist_models'

net = torch.nn.Sequential(
    ImageLifting(),
    TwoAngleConv(1, n_ch),
    torch.nn.ReLU(),

    TwoAngleResidualBlock(n_ch, n_ch),
    torch.nn.BatchNorm3d(n_ch),
    TwoAngleConv(n_ch, 2*n_ch, spatial_stride=2),
    torch.nn.ReLU(),

    TwoAngleResidualBlock(2*n_ch, 2*n_ch),
    torch.nn.BatchNorm3d(2*n_ch),
    TwoAngleConv(2*n_ch, 4*n_ch, spatial_stride=2),
    torch.nn.ReLU(),

    TwoAngleResidualBlock(4*n_ch, 4*n_ch),

    torch.nn.BatchNorm3d(4*n_ch),
    TwoAngleConv(4*n_ch, 10),
    #torch.nn.ReLU(),
    GlobalAvgPool3d()
)


from torchmetrics import Accuracy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

class InvariantModel(LightningModule):
    def __init__(self): 
        super().__init__()
        self.model = net
        self.train_accuracy = Accuracy()
        self.validation_accuracy = Accuracy()
        self.best_validation_summed_error = 1e9
        self.validation_summed_error = 0.

    def forward(self, x, batch_nb):
        x = self.model(x)
        return torch.nn.functional.log_softmax(x, dim=1)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x, batch_nb)
        loss = torch.nn.functional.nll_loss(logits, y)
        self.log('loss', loss)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log('accuracy', self.train_accuracy)
        return loss
    
    def validation_step(self,batch, batch_nb):
        x, y = batch
        print(x.shape)
        logits = self(x, batch_nb)
        loss = torch.nn.functional.nll_loss(logits, y)
        self.validation_summed_error += loss.item()
        preds = torch.argmax(logits, dim=1)
        self.validation_accuracy.update(preds, y)
    
    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.validation_accuracy.compute())
        self.validation_accuracy.reset()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.02)
        return {'optimizer': optim,
                'lr_scheduler': torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.86)}

    def on_validation_epoch_end(self):
        if self.validation_summed_error < self.best_validation_summed_error:
            torch.save(net.state_dict(), f'{save_path}/rotmnist_our_discrete_reco_{n_angles}_best.pt')
            self.best_validation_summed_error = self.validation_summed_error
        self.validation_summed_error = 0.

model = InvariantModel()
AVAIL_GPUS = min(1, torch.cuda.device_count())

# Initialize a trainer
trainer = Trainer(
    logger=TensorBoardLogger(save_dir='tmp_logs/', name=f'{save_path}/rotmnist_our_discrete_reco_{n_angles}'),
    gpus=AVAIL_GPUS,
    max_epochs=400,
    progress_bar_refresh_rate=2,
    log_every_n_steps=10,
)

val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
# Train the model ⚡
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(net.state_dict(), f'{save_path}/rotmnist_our_discrete_reco_{n_angles}.pt')