# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

from equivariant_nn_for_indirect_measurements.groups import SE2
from equivariant_nn_for_indirect_measurements.group_actions import SE2onR2, SE2onRadon
from equivariant_nn_for_indirect_measurements.conv import Conv, LocalFCNBasis
from equivariant_nn_for_indirect_measurements.networks.utils import AvgPool, BatchNorm
from equivariant_nn_for_indirect_measurements.data import FanBeamRadon
from equivariant_nn_for_indirect_measurements.networks.resnet import InvariantNet
import os
from torchvision.datasets import MNIST
from torchvision import transforms
PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")

os.environ["CUDA_VISIBLE_DEVICES"] = "6" # select GPUs to use

batch_size = 8

dist_det = 0.025
N_det = 200
D_source = 3.
D_det = 3.
beta = np.array([0., 85*np.pi/180])

im_shape = [129, 129]
im_size = [6., 6.]

from equivariant_nn_for_indirect_measurements.data.ellipse_data import ellipse_fan_radon_astra

dataset_size = 1000

train_ds, _ = ellipse_fan_radon_astra(im_shape=im_shape,
                                      im_size=im_size,
                                      D_source=D_source,
                                      D_det=D_det,
                                      dist_det=dist_det,
                                      N_det=N_det,
                                      angles=beta,
                                      n_samples=dataset_size,
                                      seed=32)

val_ds, _ = ellipse_fan_radon_astra(im_shape=im_shape,
                                      im_size=im_size,
                                      D_source=D_source,
                                      D_det=D_det,
                                      dist_det=dist_det,
                                      N_det=N_det,
                                      angles=beta,
                                      n_samples=500,
                                      seed=33)

inp_points = torch.tensor(train_ds.points[0])

print(inp_points.shape)

#train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.Compose([ZeroPad(), radon]))
#val_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.Compose([ZeroPad(), rotation_transform, radon]))

from torchmetrics import Accuracy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

net = InvariantNet(SE2onRadon,
                      radius=3.0,
                      input_points=inp_points,
                      out_classes=2,
                      knn=27,
                      initial_channels=22,
                      initial_points=2700, 
                      initial_kernel_radius=0.8,
                      downsampling_factor=2,
                      n_blocks=3)

class InvariantModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = net
        self.criterion = torch.nn.HuberLoss()
        self.best_validation_summed_error = 1e8
        self.validation_summed_error = 0.

    def forward(self, x, batch_nb):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self(x, batch_nb)
        err = torch.abs(out-y)
        loss = self.criterion(out, y)
        self.log('train/loss', loss)
        self.log('train/error0', torch.mean(err[..., 0]))
        self.log('train/error1', torch.mean(err[..., 1]))
        #self.log('train/error2', torch.mean(err[..., 2]/torch.abs(y[..., 2])))
        #self.log('train/error3', torch.mean(err[..., 3]/torch.abs(y[..., 3])))
        #self.log('train/error4', torch.mean(err[..., 4]/torch.abs(y[..., 4])))
        return loss
    
    def validation_step(self,batch, batch_nb):
        x, y = batch
        out = self(x, batch_nb)
        err = torch.abs(out-y)
        loss = self.criterion(out, y)
        self.log('val/loss', loss)
        self.log('val/mse', torch.mean(err**2))
        self.log('val/error0', torch.mean(err[..., 0]))
        self.log('val/error1', torch.mean(err[..., 1]))
        #self.log('val/error2', torch.mean(err[..., 2]/torch.abs(y[..., 2])))
        #self.log('val/error3', torch.mean(err[..., 3]/torch.abs(y[..., 3])))
        #self.log('val/error4', torch.mean(err[..., 4]/torch.abs(y[..., 4])))
        self.validation_summed_error += loss.item()
    
    def on_validation_epoch_end(self):
        if self.validation_summed_error < self.best_validation_summed_error:
            torch.save(net.state_dict(), f'paper_models/ellipse_our_{dataset_size}_best.pt')
            self.best_validation_summed_error = self.validation_summed_error
        self.validation_summed_error = 0.


    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        return {'optimizer': optim,
                'lr_scheduler': torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.96)}

model = InvariantModel()
AVAIL_GPUS = min(1, torch.cuda.device_count())

# Initialize a trainer
trainer = Trainer(
    logger=TensorBoardLogger(save_dir='tmp_logs/', name=f'paper_models/ellipse_our_{dataset_size}'),
    gpus=AVAIL_GPUS,
    max_epochs=3000,
    progress_bar_refresh_rate=2,
    log_every_n_steps=10
)

val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
# Train the model ⚡
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(net.state_dict(), f'paper_models/ellipse_our_{dataset_size}.pt')