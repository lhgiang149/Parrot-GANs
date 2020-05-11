from __future__ import print_function

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from utils import *

class Generator(nn.Module):
    def __init__(self, latent_depth, feature_depth, **kwags):
        super(Generator, self).__init__()
        self._channel = kwags.pop('channels', 3)
        # Formula to calculate output shape of ConvTranspose
        # H = (H1 - 1)*stride + HF - 2*padding
        self.model = nn.Sequential(
            # input: 1 x 100
            nn.ConvTranspose2d(latent_depth, feature_depth * 16, 4, 1, 0, bias = False),
            nn.BatchNorm2d(feature_depth * 16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # input: (feature_depth*16) x 4 x 4 
            nn.ConvTranspose2d(feature_depth * 16, feature_depth * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_depth * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # input: (feature_depth*8) x 8 x 8
            nn.ConvTranspose2d(feature_depth * 8, feature_depth * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_depth * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # input: (feature_depth*4) x 16 x 16
            nn.ConvTranspose2d(feature_depth * 4, feature_depth * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_depth * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # input: (feature_depth*2) x 32 x 32
            nn.ConvTranspose2d(feature_depth * 2, feature_depth, 4, 2 , 1, bias = Fasle),
            nn.BatchNorm2d(feature_depth),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # input: (feature_depth) x 32 x 32
            nn.ConvTranspose2d(feature_depth, self._channel, 4, 2 , 1, bias = Fasle),
            nn.Tanh()
            # final result is image 3 x 128 x 128
        )

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, feature_depth, **kwags):
        super(Discriminator, self).__init__()
        self._channel = kwags.pop('channels', 3)
        # Formula to calculate output shape of ConvTranspose
        # H = 1 + (H - HF + 2*padding)/stride
        self.model = nn.Sequential(
            # input: (nc) x 128 x 128
            nn.Conv2d(self._channel, feature_depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input: (feature_depth) x 64 x 64
            nn.Conv2d(feature_depth, feature_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input: (feature_depth*2) x 32 x 32
            nn.Conv2d(feature_depth * 2, feature_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input: (feature_depth*4) x 16 x 16
            nn.Conv2d(feature_depth * 4, feature_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # input: (feature_depth*8) x 8 x 8
            nn.Conv2d(feature_depth * 8, feature_depth * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # input: (feature_depth*8) x 4 x 4
            nn.Conv2d(feature_depth * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # final result is vector feature_depth * 16 x 1
        )

    def forward(self, input):
        return self.model(input)












