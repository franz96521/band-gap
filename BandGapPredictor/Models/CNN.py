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

class CNNModelSmall(nn.Module):
      def __init__(self, in_chanels=3, out_chanels=1):
            super(CNNModelSmall, self).__init__()
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


class CNNModelBig(nn.Module):
      def __init__(self, in_chanels=3, out_chanels=1):
            super(CNNModelBig, self).__init__()
            self.conv_layer1 = self._conv_layer_set(in_chanels, in_chanels, maxpool=False)
            self.conv_layer2 = self._conv_layer_set(in_chanels, 16)
            self.conv_layer3 = self._conv_layer_set(16, 16, maxpool=False)
            self.conv_layer4 = self._conv_layer_set(16, 32)
            self.conv_layer5 = self._conv_layer_set(32, 32, maxpool=False)
            self.conv_layer6 = self._conv_layer_set(32, 64)
            self.conv_layer7 = self._conv_layer_set(64, 64, maxpool=False)
            self.conv_layer8 = self._conv_layer_set(64, 128)
            self.conv_layer9 = self._conv_layer_set(128, 128, maxpool=False)
            self.conv_layer10 = self._conv_layer_set(128, 256)
            self.conv_layer11 = self._conv_layer_set(256, 256, maxpool=False)
            self.conv_layer12 = self._conv_layer_set(256, 512)
            self.linear1 = self._dense_layer_set(8*8*8*8, 1024)

            self.linear2 = self._dense_layer_set(1024, 512) 
            self.linear3 = self._dense_layer_set(512, 256)            
            self.linear4 = self._dense_layer_set(256, 128)
            self.linear5 = self._dense_layer_set(128, 64)
            self.linear6 = self._dense_layer_set(64, 32)
            self.linear7 = self._dense_layer_set(32, 16)
            self.linear8 = self._dense_layer_set(16, 8)
            self.linear9 = self._dense_layer_set(8, 4)
            self.linear10 = self._dense_layer_set(4, 2)
            self.linear11 = self._dense_layer_set(2, 1)
            
            
            
            
            
            
      def _conv_layer_set(self, in_c, out_c, maxpool=True):
            conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3,padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)) if maxpool else nn.Identity(),
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
            x = self.conv_layer7(x)
            x = self.conv_layer8(x)
            x = self.conv_layer9(x) 
            x = self.conv_layer10(x) 
            x = self.conv_layer11(x)
            x = self.conv_layer12(x)
            
            x = x.view(x.size(0), -1)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.linear4(x)
            x = self.linear5(x)
            x = self.linear6(x)
            x = self.linear7(x)
            x = self.linear8(x)
            x = self.linear9(x)
            x = self.linear10(x)
            x = self.linear11(x)
              
            
            return x

class BandGap(pl.LightningModule):
    def __init__(self, lr=1e-3,model=CNNModelSmall):
        super(BandGap,self).__init__()
        self.lr = lr
        self.save_hyperparameters()        
        self.loss = RMSELoss()
        self.r2score = R2Score()
        self.MAE = nn.L1Loss()
        self.model = model()
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
    