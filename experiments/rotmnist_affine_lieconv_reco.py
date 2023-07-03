# %%
from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt

from equivariant_nn_for_indirect_measurements.data.data_preparation import ParallelBeamRadon, FanBeamRadon, RadonReconstruction
from equivariant_nn_for_indirect_measurements.data.affine_mnist import MnistAffineDataset

import os
from torchvision.datasets import MNIST
from torchvision import transforms
PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")

os.environ["CUDA_VISIBLE_DEVICES"] = "4" # select GPUs to use

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


ds = MnistAffineDataset(root='/home/nick7/equivariance/equivariant-nn-for-indirect-measurements/experiments/data/', train=True, transform=transforms.Compose([ZeroPad(), reco]))
#val_ds = MnistRotDataset(root='/home/nick7/equivariance/equivariant-nn-for-indirect-measurements/experiments/data/', train=False, transform=transforms.Compose([ZeroPad(), radon]))
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = torch.utils.data.random_split(ds, [10000, 2000], generator=generator)

from lie_conv.lieConv import ImgLieResnet
from lie_conv.lieGroups import SE2

net = ImgLieResnet(num_targets=10,**{'k':128,'total_ds':.1,'fill':.1,'nbhd':25,'group':SE2(.05),'liftsamples':2})

from torchmetrics import Accuracy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

save_path = 'paper_models'

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
        logits = self(x, batch_nb)
        loss = torch.nn.functional.nll_loss(logits, y)
        self.validation_summed_error += loss.item()
        preds = torch.argmax(logits, dim=1)
        self.validation_accuracy.update(preds, y)
    
    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.validation_accuracy.compute())
        self.validation_accuracy.reset()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=3e-3, weight_decay=0.02)
        return {'optimizer': optim,
                'lr_scheduler': torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.86)}

    def on_validation_epoch_end(self):
        if self.validation_summed_error < self.best_validation_summed_error:
            torch.save(net.state_dict(), f'{save_path}/rotmnist_affine_lieconv_reco_{n_angles}_best.pt')
            self.best_validation_summed_error = self.validation_summed_error
        self.validation_summed_error = 0.

model = InvariantModel()
AVAIL_GPUS = min(1, torch.cuda.device_count())

# Initialize a trainer
trainer = Trainer(
    logger=TensorBoardLogger(save_dir='tmp_logs/', name=f'{save_path}/rotmnist_affine_lieconv_reco_{n_angles}'),
    gpus=AVAIL_GPUS,
    max_epochs=400,
    progress_bar_refresh_rate=2,
    log_every_n_steps=10,
)

val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
# Train the model ⚡
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(net.state_dict(), f'{save_path}/rotmnist_affine_lieconv_reco_{n_angles}.pt')