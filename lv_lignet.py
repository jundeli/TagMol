#######################################################
# This is a PyTorch implementation for probabilistic
# LigNet regression model.
#######################################################

import os, time
import re
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from utils import *

#############################
# Hyperparameters
#############################
lr             = 1e-5
beta1          = 0.0
beta2          = 0.9
batch_size     = 16
max_epoch      = 1000
num_workers    = 2
ligand_size    = 36
gen_dim        = 64
conv_dims      = [1024, 2048, 4096, 2048, 1024]
visualization  = False
save_step      = 100
resume_step    = 0

name = "model/lv-lignet"
log_fname = f"{name}/logs"
viz_dir = f"{name}/viz"
models_dir = f"{name}/saved_models"

if not os.path.exists(log_fname):
    os.makedirs(log_fname)
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#############################
# Define Receptor2Ligand Network
#############################
class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.activation = activation
        hidden_channels = in_channels if not hidden_channels else hidden_channels
        
        self.c1 = nn.Conv3d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv3d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c1 = nn.utils.spectral_norm(self.c1)
        self.c2 = nn.utils.spectral_norm(self.c2)

    def forward(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

class LV_LigNet(nn.Module):
    """Deep probabilistic regression for mapping receptors to ligand distributions."""
    def __init__(self, conv_dims, in_channels=8, out_channels=3, activation=nn.ReLU()):
        super(LV_LigNet, self).__init__()