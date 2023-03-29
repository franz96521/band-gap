from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import path
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

class CNNModel(nn.Module):
      def __init__(self, in_chanels=3, out_chanels=1):
            super(CNNModel, self).__init__()
            self.conv_layer1 = self._conv_layer_set(in_chanels, 16)
            self.conv_layer2 = self._conv_layer_set(16, 32)
            self.conv_layer3 = self._conv_layer_set(32, 64)
            self.conv_layer4 = self._conv_layer_set(64, 128)
            self.conv_layer5 = self._conv_layer_set(128, 256)
            self.conv_layer6 = self._conv_layer_set(256, 512)            
            self.linear1 = self._dense_layer_set(8*8*8*8, 1024)
            self.linear2 = self._dense_layer_set(1024, 512) 
            self.linear3 = self._dense_layer_set(512, 256)            
            self.linear4 = self._dense_layer_set(256, 128)
            self.linear5 = self._dense_layer_set(128, 64)
            self.linear6 = self._dense_layer_set(64, 1)
            
            
      def _conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3,padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
            # batch norm 3d 
            
            )
            return conv_layer
      def _dense_layer_set(self, in_c, out_c,drop= 0.2):
            dense_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_c),
            nn.Dropout(p=drop)            
            )
            return dense_layer

      def forward(self, x):
            x = self.conv_layer1(x)
            x = self.conv_layer2(x)
            x = self.conv_layer3(x)
            x = self.conv_layer4(x)
            x = self.conv_layer5(x)
            x = self.conv_layer6(x)
            x = x.view(x.size(0), -1)    
            
            # print(x.shape)        
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.linear4(x)
            x = self.linear5(x)
            x = self.linear6(x)
            return x

class BandGap(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(BandGap,self).__init__()
        self.lr = lr
        self.save_hyperparameters()        
        self.loss = RMSELoss()
        self.r2score = R2Score()
        self.model = CNNModel()
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
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_r2", r2, prog_bar=True, on_step=True)
        return {"loss": loss, "log": {"train_loss": loss, "train_r2": r2}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        r2 = self.r2score(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log("val_r2", r2, prog_bar=True, on_step=True)
        return {"loss": loss, "log": {"val_loss": loss, "val_r2": r2}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        r2 = self.r2score(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_step=True)
        self.log("test_r2", r2, prog_bar=True, on_step=True)
        return {"loss": loss, "log": {"test_loss": loss, "test_r2": r2}}
    