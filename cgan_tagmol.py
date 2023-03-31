###################################################################
# This is an official PyTorch implementation for Target-specific
# Generation of Molecules (TagMol)
# Author: Junde Li, The Pennsylvania State University
# Date: Mar 25, 2023
###################################################################

import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from rdkit import Chem
from dataloader import PDBbindPLDataset, Normalize, RandomRotateJitter, ToTensor
from model import PointNetEncoder, Generator, Discriminator, m_Discriminator
import itertools
from utils import *
from frechetdist import frdist
import csv

import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Hyperparameters
# --------------------------
lr             = 1e-4
batch_size     = 32
max_epoch      = 2000
num_workers    = 2
ligand_size    = 14
x_dim          = 16 # standard GAN
z_dim          = 16
n_pc_points    = 4096
g_conv_dims    = [64, 256, 1024]
d_conv_dim     = [[128, 64], 128, [128, 64]]
node_dim       = 64
n_atom_types   = 7
n_bond_types   = 5
dropout        = 0.2

dataset        = 'refined'
n_critic       = 5
lambda_gp      = 10
alpha_l2       = 1e-3
beta_le        = 1e-5
gamma_lr       = 1e-4
save_step      = 500

atom_decoder   = {0: 0, 1: 6, 2: 7, 3: 8, 4: 9, 5: 16, 6:17}
bond_decoder   = {0: Chem.rdchem.BondType.ZERO,
                  1: Chem.rdchem.BondType.SINGLE,
                  2: Chem.rdchem.BondType.DOUBLE,
                  3: Chem.rdchem.BondType.TRIPLE,
                  4: Chem.rdchem.BondType.AROMATIC}

name           = f"model/tagmol-ligsize/{ligand_size}"
log_dir        = f"{name}"
models_dir     = f"{name}/{dataset}_saved_models"
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda           = True if torch.cuda.is_available() else False
Tensor         = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


proteins = pickle.load(open(f'data/pdbbind/{dataset}-set/proteins.p', "rb"))
atoms = pickle.load(open(f'data/pdbbind/{dataset}-set/atoms.p', "rb"))
bonds = pickle.load(open(f'data/pdbbind/{dataset}-set/bonds.p', "rb"))

train_dataset = Data.TensorDataset(proteins, atoms, bonds)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

# Initialize models and make optimizers.
encoder       = PointNetEncoder(x_dim, channel=4, feature_transform=True)
generator     = Generator(x_dim, z_dim, g_conv_dims, ligand_size, n_atom_types, n_bond_types)
discriminator = m_Discriminator(d_conv_dim, n_atom_types, n_bond_types, dropout)
# discriminator = Discriminator(c_in=n_atom_types, c_out=node_dim, c_hidden=32, n_relations=n_bond_types, n_layers=3)  

opt_gen       = torch.optim.Adam(generator.parameters(), lr, (0.9, 0.999))
opt_disc      = torch.optim.Adam(discriminator.parameters(), lr, (0.9, 0.999))

if cuda:
    encoder.cuda()
    generator.cuda()
    discriminator.cuda()

start_epoch = 0
if start_epoch:
    for model in [generator, discriminator]:
        path = os.path.join(models_dir, f'{ligand_size}-{start_epoch}-{str(type(model)).split(".")[-1][:-2]}.ckpt')
        model.load_state_dict(torch.load(path))
        model.train()
    print(f'Loaded model checkpoints from {models_dir}.')

def main():
    # train loop
    print('Start traning...')

    for epoch in tqdm(range(start_epoch, max_epoch),  desc='total progress'):
        losses_D  = []
        losses_G  = []
        fd_scores = []
        cur_time = time.strftime("%D %H:%M:%S", time.localtime())
        epoch_log = f"{cur_time}\tepoch {epoch+1}\t"

        for batch, sample_batched in enumerate(train_loader):
            protein = sample_batched[0]
            r_atoms, r_bonds = sample_batched[1:]
            batch_log = f"{epoch+1}:{batch}\n"

            bs = protein.size(0)
            
            # -----------------------
            #  Train Discriminator
            # -----------------------
            opt_disc.zero_grad()

            # Encode protein features using encoder.
            x = encoder(protein.transpose(2, 1))
            # Sample noise as generator input.
            z = Variable(Tensor(np.random.normal(0, 1, (bs, z_dim))))
            f_atoms, f_bonds = generator(z, x)
            # Hard categorical sampling fake ligands from probabilistic distribution.
            f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
            f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

            # Validity for real and fake samples.
            r_validity = discriminator(r_bonds, None, r_atoms)
            f_validity = discriminator(f_bonds, None, f_atoms)

            # Calculate adient penalty.
            gradient_penalty = compute_gradient_penalty(discriminator, r_atoms, r_bonds, f_atoms, f_bonds, True)

            # Adversarial loss plus gradient penalty.
            loss_D = -torch.mean(r_validity) + torch.mean(f_validity) + lambda_gp * gradient_penalty
            losses_D.append(loss_D.item())

            loss_D.backward()
            opt_disc.step()

            batch_log += f"loss_d: {-torch.mean(r_validity).item():.2f}\t" + \
                            f"{torch.mean(f_validity).item():.2f}\t{lambda_gp * gradient_penalty.item():.2f}\n"

            # Train other networks every n_critic steps
            if (batch+1) % n_critic == 0:

                # -------------------------------
                #  Train Generator and Encoder
                # -------------------------------
                opt_gen.zero_grad()
                
                if x_dim:
                    x = encoder(protein.transpose(2, 1))
                z = Variable(Tensor(np.random.normal(0, 1, (batch_size, z_dim))))
                f_atoms, f_bonds = generator(z, x)
                # Hard categorical sampling fake ligands from probabilistic distribution.
                f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
                f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

                # Validity for fake samples.
                f_validity = discriminator(f_bonds, None, f_atoms)
                loss_G_fake = - torch.mean(f_validity)

                loss_G = loss_G_fake
                losses_G.append(loss_G.item())

                loss_G.backward()
                opt_gen.step()

                r_dist =[list(r_atoms[i].reshape(-1).cpu().detach().numpy()) \
                                    + list(r_bonds[i].reshape(-1).cpu().detach().numpy()) for i in range(batch_size)]
                f_dist =[list(f_atoms[i].reshape(-1).cpu().detach().numpy()) \
                                    + list(f_bonds[i].reshape(-1).cpu().detach().numpy()) for i in range(batch_size)]
                fd_bond_atom = frdist(r_dist, f_dist)
                fd_scores.append(fd_bond_atom)

                # Saving model checkpoint with lowest FD score
                if "fd_bond_atom_min" not in locals():
                    fd_bond_atom_min = 30
                if fd_bond_atom_min > fd_bond_atom:
                    if "lowest_ind" not in locals():
                        lowest_ind = 0

                    if lowest_ind:
                        for model in [encoder, generator, discriminator]:
                            os.remove(os.path.join(models_dir, \
                                f'{ligand_size}-{lowest_ind}-{str(type(model)).split(".")[-1][:-2]}.ckpt'))

                    lowest_ind = epoch+1
                    fd_bond_atom_min = fd_bond_atom

                    for model in [encoder, generator, discriminator]:
                        path = os.path.join(models_dir, \
                            f'{ligand_size}-{epoch+1}-{str(type(model)).split(".")[-1][:-2]}.ckpt')
                        torch.save(model.state_dict(), path)
                        
                    with open(os.path.join(models_dir, 'lowest_indices.csv'), 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch+1] + [fd_bond_atom])

                batch_log += f"loss_g: {loss_G_fake.item():.2f}\n" 

                print_and_save(batch_log, f"{log_dir}/batch-log.txt")
        epoch_log += f"loss_d:{np.mean(losses_D):.4f}\t loss_g:{np.mean(losses_G):.4f}\t fd:{np.mean(fd_scores):.4f}\t"
        print_and_save(epoch_log, f"{log_dir}/epoch-log.txt")

        # Save model checkpoints.
        if (epoch+1) % save_step == 0:
            for model in [encoder, generator, discriminator]:
                path = os.path.join(models_dir, f'{ligand_size}-{epoch+1}-{str(type(model)).split(".")[-1][:-2]}.ckpt')
                torch.save(model.state_dict(), path)
            print(f'Saved model checkpoints into {models_dir}...')


if __name__ == '__main__':
    main()