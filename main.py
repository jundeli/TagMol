###################################################################
# This is an official PyTorch implementation for Target-specific
# Generation of Molecules (TagMol)
# Author: Junde Li, The Pennsylvania State University
# Date: Aug 1, 2022
###################################################################

import os, time
import re
from sklearn import datasets
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from rdkit import Chem
from dataloader import PDBbindPLDataset, Normalize, RandomRotateJitter, ToTensor
from model import PointNetEncoder, Generator, Discriminator, EnergyModel, RewardModel
import itertools


# --------------------------
# Hyperparameters
# --------------------------
lr             = 1e-4
batch_size     = 16
max_epoch      = 400
num_workers    = 2
ligand_size    = 32
x_dim          = 512
z_dim          = 128
n_pc_points    = 4096
conv_dims      = [1024, 2048, 2048, 1024]
node_dim       = 64
n_atom_types   = 7
n_bond_types   = 5
dataset        = 'tiny'
save_step      = 100

name           = f"model/tagmol"
log_dir        = f"{name}"
models_dir     = f"{name}/{dataset}_saved_models"
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda           = True if torch.cuda.is_available() else False

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


# --------------------------
# Make dataloaders
# --------------------------
train_dataset = PDBbindPLDataset(root_dir=f'data/pdbbind/{dataset}-set',
                                n_points=n_pc_points,
                                lig_size=ligand_size,
                                train=True,
                                transform=transforms.Compose([
                                    Normalize(),
                                    RandomRotateJitter(sigma=0.15),
                                    ToTensor()
                                ]))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True)

test_dataset = PDBbindPLDataset(root_dir=f'data/pdbbind/{dataset}-set',
                                n_points=n_pc_points,
                                lig_size=ligand_size,
                                train=False,
                                transform=transforms.Compose([
                                    Normalize(),
                                    RandomRotateJitter(sigma=0.15),
                                    ToTensor()
                                ]))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=True)


# -----------------------------
# Make models and optimizers
# -----------------------------
encoder       = torch.nn.DataParallel(PointNetEncoder(x_dim, channel=4, feature_transform=True))
generator     = torch.nn.DataParallel(Generator(x_dim, z_dim, conv_dims, ligand_size, (n_atom_types, n_bond_types)))
discriminator = torch.nn.DataParallel(Discriminator(c_in=n_atom_types, c_out=node_dim, c_hidden=32, n_relations=n_bond_types, n_layers=3))
energy_model  = torch.nn.DataParallel(EnergyModel(x_dim, c_in=n_atom_types, c_out=node_dim, n_relations=n_bond_types, n_layers=3))
reward_model  = torch.nn.DataParallel(RewardModel(c_in=n_atom_types, c_out=node_dim, c_hidden=32, n_relations=n_bond_types, n_layers=3))

opt_enc_gen   = torch.optim.Adam(itertools.chain(encoder.parameters(), generator.parameters()), lr, (0.0, 0.9))
opt_disc      = torch.optim.Adam(discriminator.parameters(), lr, (0.0, 0.9))
opt_ene       = torch.optim.Adam(energy_model.parameters(), lr, (0.0, 0.9))
opt_rew       = torch.optim.Adam(reward_model.parameters(), lr, (0.0, 0.9))


