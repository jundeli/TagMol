###################################################################
# This is an official PyTorch implementation for Target-specific
# Generation of Molecules (TagMol)
# Author: Junde Li, The Pennsylvania State University
# Date: Aug 1, 2022
###################################################################

import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from rdkit import Chem
from dataloader import PDBbindPLDataset, Normalize, RandomRotateJitter, ToTensor
from model import PointNetEncoder, Generator, Discriminator, EnergyModel, RewardModel
import itertools
from utils import *


# --------------------------
# Hyperparameters
# --------------------------
lr             = 1e-4
batch_size     = 16
max_epoch      = 200
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
n_critic       = 5
lambda_gp      = 10
alpha_l2       = 0.1
save_step      = 100

atom_decoder = {0: 0, 1: 6, 2: 7, 3: 8, 4: 9, 5: 16, 6:17}
bond_decoder = {0: Chem.rdchem.BondType.ZERO,
                1: Chem.rdchem.BondType.SINGLE,
                2: Chem.rdchem.BondType.DOUBLE,
                3: Chem.rdchem.BondType.TRIPLE,
                4: Chem.rdchem.BondType.AROMATIC
                }

name           = f"model/tagmol"
log_dir        = f"{name}"
models_dir     = f"{name}/{dataset}_saved_models"
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda           = True if torch.cuda.is_available() else False
Tensor         = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


# Make dataloaders
train_dataset = PDBbindPLDataset(root_dir=f'data/pdbbind/{dataset}-set',
                                n_points=n_pc_points,
                                lig_size=ligand_size,
                                train=True,
                                transform=transforms.Compose([
                                    Normalize(),
                                    RandomRotateJitter(sigma=0.15),
                                    ToTensor()
                                ]))
train_loader = DataLoader(train_dataset, batch_size=batch_size,
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
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=True)


# Initialize models and make optimizers.
encoder       = torch.nn.DataParallel(PointNetEncoder(x_dim, channel=4, feature_transform=True))
generator     = torch.nn.DataParallel(Generator(x_dim, z_dim, conv_dims, ligand_size, (n_atom_types, n_bond_types)))
discriminator = torch.nn.DataParallel(Discriminator(c_in=n_atom_types, c_out=node_dim, c_hidden=32, n_relations=n_bond_types, n_layers=3))
energy_model  = torch.nn.DataParallel(EnergyModel(x_dim, c_in=n_atom_types, c_out=node_dim, n_relations=n_bond_types, n_layers=3))
reward_model  = torch.nn.DataParallel(RewardModel(c_in=n_atom_types, c_out=node_dim, c_hidden=32, n_relations=n_bond_types, n_layers=3))

opt_enc_gen   = torch.optim.Adam(itertools.chain(encoder.parameters(), generator.parameters()), lr, (0.0, 0.9))
opt_disc      = torch.optim.Adam(discriminator.parameters(), lr, (0.0, 0.9))
opt_ene       = torch.optim.Adam(energy_model.parameters(), lr, (0.0, 0.9))
opt_rew       = torch.optim.Adam(reward_model.parameters(), lr, (0.0, 0.9))

if cuda:
    encoder.cuda()
    generator.cuda()
    discriminator.cuda()
    energy_model.cuda()
    reward_model.cuda()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

# Initialize model weights.
encoder.apply(weights_init_normal)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
energy_model.apply(weights_init_normal)
reward_model.apply(weights_init_normal)

def main():
    # train loop
    print('Start traning...')

    for epoch in tqdm(range(max_epoch),  desc='total progress'):
        losses = []
        for batch, sample_batched in enumerate(train_loader):
            protein = sample_batched['protein']
            r_atoms, r_bonds = sample_batched['ligand']
            curr_log = f"epoch {epoch+1}\t"
            
            # -----------------------
            #  Train Discriminator
            # -----------------------
            opt_disc.zero_grad()

            # Encode protein features using encoder.
            x = encoder(protein.transpose(2, 1))
            # Sample noise as generator input.
            z = Variable(Tensor(np.random.normal(0, 1, (batch_size, z_dim))))
            f_atoms, f_bonds = generator(x, z)
            # Hard categorical sampling fake ligands from probabilistic distribution.
            f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
            f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

            # Validity for real and fake samples.
            r_validity = discriminator(r_atoms, r_bonds)
            f_validity = discriminator(f_atoms, f_bonds)

            # Calculate adient penalty.
            gradient_penalty = compute_gradient_penalty(discriminator, r_atoms, r_bonds, f_atoms, f_bonds)

            # Adversarial loss
            loss_D = -torch.mean(r_validity) + torch.mean(f_validity) + lambda_gp * gradient_penalty

            loss_D.backward()
            opt_disc.step()

        # Train other networks every n_critic steps
        if (batch+1) % n_critic == 0:
            
            # -------------------
            #  Train Generator
            # -------------------
            opt_enc_gen.zero_grad()

            z = Variable(Tensor(np.random.normal(0, 1, (batch_size, z_dim))))
            f_atoms, f_bonds = generator(x, z)
            # Hard categorical sampling fake ligands from probabilistic distribution.
            f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
            f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

            # Validity for fake samples.
            f_validity = discriminator(f_atoms, f_bonds)
            loss_G_fake = - torch.mean(f_validity)

            # Energies for real and fake samples.
            r_out = energy_model(x, r_atoms, r_bonds)
            f_out = energy_model(x, f_atoms, f_bonds)
            loss_E = r_out.mean() - f_out.mean() + alpha_l2 * (r_out ** 2 + f_out ** 2).mean()

            # Properties for real and fake ligands.
            r_pred_properties = reward_model(r_atoms, r_bonds)
            f_pred_properties = reward_model(f_atoms, f_bonds)



