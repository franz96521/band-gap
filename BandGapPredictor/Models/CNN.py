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
            self.linear11 = self._dense_layer_set(2, out_chanels)
            
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
        
class DenseSkip(nn.Module):
    def __init__(self, in_chanels=3, out_chanels=1,yshape = 128,droprate = 0.2):
        super(DenseSkip, self).__init__()        
        self.linear1 = nn.Linear(in_chanels+yshape, out_chanels)
        self.relu1 =  nn.LeakyReLU()
        self.batchnorm1 =nn.BatchNorm1d(out_chanels)
        self.drop =  nn.Dropout(p=droprate) 
    def forward(self, x,y ):
        # concat x and y
        # print("x ",x.shape)
        # print(y.shape)
        y = y.view(y.size(0), -1)
        # print("y ",y.shape)
        c = torch.cat((x,y),dim=1)
        # print(c.shape)
        x = self.linear1(c)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.drop(x)
        return x
        
        
class CNNModelBigSkip(nn.Module):
      def __init__(self, in_chanels=3, out_chanels=1):
            super(CNNModelBigSkip, self).__init__()
            # add skip connection
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
            self.linear1 = self._dense_layer_set(8*8*8*8, 1024,8*8*8*8)
            self.linear2 = self._dense_layer_set(1024, 512,16384)
            self.linear3 = self._dense_layer_set(512, 256,16384)
            self.linear4 = self._dense_layer_set(256, 128,65536)
            self.linear5 = self._dense_layer_set(128, 64,65536)
            self.linear6 = self._dense_layer_set(64, 32,262144)
            self.linear7 = self._dense_layer_set(32, 16,262144)
            self.linear8 = self._dense_layer_set(16, 8,1048576)
            self.linear9 = self._dense_layer_set(8, 4,1048576)
            self.linear10 = self._dense_layer_set(4, 2,4194304)
            self.linear11 = self._dense_layer_set(2, out_chanels,4194304)
            
      def _conv_layer_set(self, in_c, out_c, maxpool=True):
          

            conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3,padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)) if maxpool else nn.Identity(),
            # batch norm 3d 
            
            )
            return conv_layer
      def _dense_layer_set(self, in_c, out_c,yshape=0,drop= 0.2):
            dense_layer = DenseSkip(in_c, out_c,yshape,drop)
            return dense_layer

      def forward(self, x):
            x1 = self.conv_layer1(x)
            
            x2 = self.conv_layer2(x1)
            x3 = self.conv_layer3(x2)
            x4 = self.conv_layer4(x3)
            x5 = self.conv_layer5(x4)
            x6 = self.conv_layer6(x5)
            x7 = self.conv_layer7(x6)
            x8 = self.conv_layer8(x7)
            x9 = self.conv_layer9(x8)
            x10 = self.conv_layer10(x9)
            x11 = self.conv_layer11(x10)
            x12 = self.conv_layer12(x11)
            
            x13 = x12.view(x12.size(0), -1)
            # print(x12.shape)
            x14 = self.linear1(x13,x12)
            x15 = self.linear2(x14,x11)
            x16 = self.linear3(x15,x10)
            x17 = self.linear4(x16,x9)
            x18 = self.linear5(x17,x8)
            x19 = self.linear6(x18,x7)
            x20 = self.linear7(x19,x6)
            x21 = self.linear8(x20,x5)
            x22 = self.linear9(x21,x4)
            x23 = self.linear10(x22,x3)
            x24 = self.linear11(x23,x2)
            return x24
            

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
    