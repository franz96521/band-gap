from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import path
import sys
import os 
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)

from LossFunctions.lossFunctions import RMSELoss

import torch
import torch.nn as nn

from torchmetrics import R2Score

import pytorch_lightning as pl
class BandGap(pl.LightningModule):
    def __init__(self, lr=1e-3,model=None):
        super(BandGap,self).__init__()
        self.lr = lr
        self.save_hyperparameters()        
        self.loss = RMSELoss()
        self.r2score = R2Score()
        self.MAE = nn.L1Loss()
        self.model = model
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "val_loss",                
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        r2 = self.r2score(y_hat, y)
        mae = self.MAE(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_r2", r2, prog_bar=True, on_step=True)
        self.log("train_MAE",mae , prog_bar=True, on_step=True)
        return {"loss": loss, "log": {"train_loss": loss, "train_r2": r2}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        r2 = self.r2score(y_hat, y)
        mae = self.MAE(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log("val_r2", r2, prog_bar=True, on_step=True)
        self.log("val_MAE",mae , prog_bar=True, on_step=True)
        return {"loss": loss, "log": {"val_loss": loss, "val_r2": r2}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        r2 = self.r2score(y_hat, y)
        mae = self.MAE(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_step=True)
        self.log("test_r2", r2, prog_bar=True, on_step=True)
        self.log("test_MAE",mae , prog_bar=True, on_step=True)
        return {"loss": loss, "log": {"test_loss": loss, "test_r2": r2}}
    