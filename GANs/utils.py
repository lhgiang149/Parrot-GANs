import os
import random
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Conv':
        nn.init.kaiming_normal_(m.weight.data, mode = 'fan_in', nonlinearity= 'leaky_relu')
    elif classname == 'BatchNorm':
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data,0)
