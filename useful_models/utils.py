import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct


class Embedding(nn.Module):
    # in_channel = 1 (since 20 bus lines are extracted, combine them into 1 image), code_dim is the dim of embedded bus code
    def __init__(self, code_dim):
        super(Embedding, self).__init__()
        # input: (batch, 1, 50, 50).  output: (batch, 64, 23, 23)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=6, stride=2, padding=0)
        # output: (batch, 128, 11, 11)
        self.conv2 = nn.Conv2d(64, 128, 7, 2, 2)
        # output: (batch, 1024, 5, 5)
        self.conv3 = nn.Conv2d(128, 1024, 7, 2, 2)

        # input: (batch, 1024, 5, 5).  output: (batch, 128, 1, 1)
        self.conv4 = nn.Conv2d(1024, 128, 5, 2, 0)

        # output: (batch, code_dim)
        self.linear1 = nn.Linear(128, code_dim)

    # x shape: (batch, 20, 50, 50)
    def forward(self, x):
        x = x.view(x.size(0), 1, 20, 50, 50).sum(2)

        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv4(x), 0.1, inplace=True)
        x = x.view(x.size(0), -1)
        out = torch.tanh(self.linear1(x))

        return out      # out shape: (batch, code_dim)


class Generator(nn.Module):
    def __init__(self, noise_dim, code_dim):
        super().__init__()
        # input: (batch, 100+code_dim, 1, 1).  output: (batch, 1024, 5, 5)
        self.tconv1 = nn.ConvTranspose2d(noise_dim + code_dim, out_channels=1024, kernel_size=5, stride=2, padding=0)

        # output: (batch, 128, 11, 11)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, kernel_size=7, stride=2, padding=2)

        # output: (batch, 64, 23, 23)
        self.tconv3 = nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, padding=2)

        # output: (batch, 1, 50, 50)
        self.tconv4 = nn.ConvTranspose2d(64, 1, kernel_size=6, stride=2, padding=0)

    def forward(self, x, c):
        x = torch.cat([x, c], 1).view(x.size(0), -1, 1, 1)

        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))

        img = torch.tanh(self.tconv4(x))

        return img


class Discriminator(nn.Module):
    def __init__(self, code_dim):
        super().__init__()

        # input: (batch, 1, 50, 50).  output: (batch, 64, 23, 23)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=6, stride=2, padding=0)
        # output: (batch, 128, 11, 11)
        self.conv2 = nn.Conv2d(64, 128, 7, 2, 2)
        # output: (batch, 1024, 5, 5)
        self.conv3 = nn.Conv2d(128, 1024, 7, 2, 2)

        # input: (batch, 1024, 5, 5).  output: (batch, 128, 1, 1)
        self.conv4 = nn.Conv2d(1024, 128, 5, 2, 0)

        self.linear1 = nn.Linear(128, code_dim)

        # input: (batch, 1, 50, 50).  output: (batch, 64, 23, 23)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=6, stride=2, padding=0)
        # output: (batch, 128, 11, 11)
        self.conv6 = nn.Conv2d(64, 128, 7, 2, 2)
        # output: (batch, 1024, 5, 5)
        self.conv7 = nn.Conv2d(128, 1024, 7, 2, 2)

        # input: (batch, 1024, 5, 5).  output: (batch, 128, 1, 1)
        self.conv8 = nn.Conv2d(1024, 128, 5, 2, 0)

        self.linear2 = nn.Linear(128 + code_dim, 128)

    def forward(self, x, bus):
        bus = bus.view(bus.size(0), 1, 20, 50, 50).sum(2)
        bus = F.leaky_relu(self.conv1(bus), 0.1, inplace=True)
        bus = F.leaky_relu(self.conv2(bus), 0.1, inplace=True)
        bus = F.leaky_relu(self.conv3(bus), 0.1, inplace=True)
        bus = F.leaky_relu(self.conv4(bus), 0.1, inplace=True)
        bus = bus.view(bus.size(0), -1)
        bus = torch.tanh(self.linear1(bus))

        x = F.leaky_relu(self.conv5(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv6(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv7(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv8(x), 0.1, inplace=True)

        x = torch.cat([x.view(x.size(0), -1), bus], 1).view(x.size(0), -1)
        x = torch.tanh(self.linear2(x))

        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # output: (batch, 128)
        x = x.view(x.size(0), -1)
        # output: (batch, 128+code_dim)
        # x = torch.cat([x, c], 1)
        output = torch.sigmoid(self.fc(x))     # output: (batch, 1)

        return output


class QHead(nn.Module):
    def __init__(self, mu_dim, var_dim):
        super().__init__()
        self.fc = nn.Linear(128, 1024)
        # self.bn = nn.BatchNorm2d(1024),

        #self.fc_continuous = nn.Linear(1024, code_dim)
        self.fc_mu = nn.Linear(1024, mu_dim)
        self.fc_var = nn.Linear(1024, var_dim)

    def forward(self, x):
        # output: (batch, 128)
        x = x.view(x.size(0), -1)
        # output: (batch, 1024)
        x = F.leaky_relu(self.fc(x), 0.1, inplace=True)

        # cont_logits = self.fc_continuous(x)

        mu = self.fc_mu(x)
        var = torch.exp(self.fc_var(x))

        return mu, var


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll