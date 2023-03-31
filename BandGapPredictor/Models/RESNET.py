import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import torchvision
class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Resnet, self).__init__()
        
        self.resnet = torchvision.models.video.r3d_18(pretrained=False)
        self.linear1 = nn.Linear(400, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.resnet(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x
