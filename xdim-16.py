###################################################################
# This is an official PyTorch implementation for Target-specific
# Generation of Molecules (TagMol)
# Author: Junde Li, The Pennsylvania State University
# Date: Aug 1, 2022
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from rdkit import Chem

import sys
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current))

from dataloader import PDBbindPLDataset, Normalize, RandomRotateJitter, ToTensor
from model import PointNetEncoder, Generator, Discriminator, EnergyModel, RewardModel
import itertools
from utils import *

import warnings
warnings.filterwarnings("ignore")


# --------------------------
# Hyperparameters
# --------------------------
lr             = 3e-5
batch_size     = 16
max_epoch      = 600
num_workers    = 2
ligand_size    = 14
x_dim          = 16
z_dim          = 64
n_pc_points    = 4096
conv_dims      = [64, 256, 1024]
node_dim       = 64
n_atom_types   = 7
n_bond_types   = 5

dataset        = 'refined'
n_critic       = 5
lambda_gp      = 10
alpha_l2       = 1e-3
beta_le        = 1e-5
gamma_lr       = 1e-4
save_step      = 100

atom_decoder   = {0: 0, 1: 6, 2: 7, 3: 8, 4: 9, 5: 16, 6:17}
bond_decoder   = {0: Chem.rdchem.BondType.ZERO,
                  1: Chem.rdchem.BondType.SINGLE,
                  2: Chem.rdchem.BondType.DOUBLE,
                  3: Chem.rdchem.BondType.TRIPLE,
                  4: Chem.rdchem.BondType.AROMATIC}

name           = f"tagmol-xdim/{x_dim}"
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
                        shuffle=True, drop_last=False)

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
                        shuffle=True, drop_last=False)


# Initialize models and make optimizers.
encoder       = PointNetEncoder(x_dim, channel=4, feature_transform=True)
generator     = Generator(x_dim, z_dim, conv_dims, ligand_size, n_atom_types, n_bond_types)
discriminator = Discriminator(c_in=n_atom_types, c_out=node_dim, c_hidden=32, n_relations=n_bond_types, n_layers=3)
energy_model  = EnergyModel(x_dim, c_in=n_atom_types, c_out=node_dim, n_relations=n_bond_types, n_layers=3)
reward_model  = RewardModel(c_in=n_atom_types, c_out=node_dim, c_hidden=32, n_relations=n_bond_types, n_layers=3)

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


start_epoch = 300
for model in [encoder, generator, discriminator]: #, energy_model, reward_model
    path = os.path.join(models_dir, f'{ligand_size}-{start_epoch}-{str(type(model)).split(".")[-1][:-2]}.ckpt')
    model.load_state_dict(torch.load(path))
    model.eval()
print(f'Loaded model checkpoints from {models_dir}...')


def main():
    # train loop
    print('Start traning...')

    for epoch in tqdm(range(start_epoch, max_epoch),  desc='total progress'):
        losses_D = []
        losses_G = []
        losses_E = []
        losses_R = []
        cur_time = time.strftime("%D %H:%M:%S", time.localtime())
        epoch_log = f"{cur_time}\tepoch {epoch+1}\t"

        for batch, sample_batched in enumerate(train_loader):
            protein = sample_batched['protein']
            r_atoms, r_bonds = sample_batched['ligand']
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
            f_atoms, f_bonds = generator(x, z)
            # Hard categorical sampling fake ligands from probabilistic distribution.
            f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
            f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

            # Validity for real and fake samples.
            r_validity = discriminator((r_atoms, r_bonds))
            f_validity = discriminator((f_atoms, f_bonds))

            # Calculate adient penalty.
            gradient_penalty = compute_gradient_penalty(discriminator, r_atoms, r_bonds, f_atoms, f_bonds)

            # Adversarial loss plus gradient penalty.
            loss_D = -torch.mean(r_validity) + torch.mean(f_validity) + lambda_gp * gradient_penalty
            losses_D.append(loss_D.item())

            loss_D.backward()
            opt_disc.step()

            batch_log += f"loss_d: {-torch.mean(r_validity).item():.2f}\t{torch.mean(f_validity).item():.2f}\t{lambda_gp * gradient_penalty.item():.2f}\n"

            # Train other networks every n_critic steps
            if (batch+1) % n_critic == 0:

                # -------------------------------
                #  Train Generator and Encoder
                # -------------------------------
                opt_enc_gen.zero_grad()
                
                x = encoder(protein.transpose(2, 1))
                z = Variable(Tensor(np.random.normal(0, 1, (batch_size, z_dim))))
                f_atoms, f_bonds = generator(x, z)
                # Hard categorical sampling fake ligands from probabilistic distribution.
                f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
                f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

                # Validity for fake samples.
                f_validity = discriminator((f_atoms, f_bonds))
                loss_G_fake = - torch.mean(f_validity)

                # # Energies for real and fake samples.
                # r_out = energy_model(x, r_atoms, r_bonds)
                # f_out = energy_model(x, f_atoms, f_bonds)
                # loss_E = torch.mean(r_out) - torch.mean(f_out) + alpha_l2 * torch.mean(r_out ** 2 + f_out ** 2)

                # # Properties for real and fake ligands.
                # r_pred_properties = reward_model(r_atoms, r_bonds)
                # f_pred_properties = reward_model(f_atoms, f_bonds)

                # # Get rdkit evaluated property scores.
                # r_properties, f_properties = compute_rdkit_property(r_atoms, r_bonds, f_atoms, f_bonds)

                # loss_R = torch.mean((r_pred_properties - r_properties)**2 + \
                #                                 (f_pred_properties - f_properties)**2)

                loss_G = loss_G_fake #+ beta_le*loss_E + gamma_lr*loss_R
                losses_G.append(loss_G.item())

                loss_G.backward()
                opt_enc_gen.step()

                batch_log += f"loss_g: {loss_G_fake.item():.2f}\n" #\t{loss_E.item():.2f}\t{loss_R.item():.2f}

                # # ------------------------
                # #  Train Energy Network
                # # ------------------------
                # opt_ene.zero_grad()

                # x = encoder(protein.transpose(2, 1))
                # f_atoms, f_bonds = generator(x, z)
                # # Hard categorical sampling fake ligands from probabilistic distribution.
                # f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
                # f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

                # # Energies for real and fake samples.
                # r_out = energy_model(x, r_atoms, r_bonds)
                # f_out = energy_model(x, f_atoms, f_bonds)
                # loss_E = torch.mean(r_out) - torch.mean(f_out) + alpha_l2 * torch.mean(r_out ** 2 + f_out ** 2)
                # losses_E.append(loss_E.item())

                # loss_E.backward()
                # opt_ene.step()

                # batch_log += f"loss_e: {torch.mean(r_out).item():.2f}\t{- torch.mean(f_out).item():.2f}\t{alpha_l2 * torch.mean(r_out ** 2 + f_out ** 2).item():.2f}\n"


                # # ------------------------
                # #  Train Reward Network
                # # ------------------------
                # opt_rew.zero_grad()

                # x = encoder(protein.transpose(2, 1))
                # f_atoms, f_bonds = generator(x, z)
                # # Hard categorical sampling fake ligands from probabilistic distribution.
                # f_atoms = F.gumbel_softmax(f_atoms, tau=1, hard=True)
                # f_bonds = F.gumbel_softmax(f_bonds, tau=1, hard=True)

                # # Properties for real and fake ligands.
                # r_pred_properties = reward_model(r_atoms, r_bonds)
                # f_pred_properties = reward_model(f_atoms, f_bonds)

                # # Get rdkit evaluated property scores.
                # r_properties, f_properties = compute_rdkit_property(r_atoms, r_bonds, f_atoms, f_bonds)

                # loss_R = torch.mean((r_pred_properties - r_properties)**2 + \
                #                                 (f_pred_properties - f_properties)**2)
                # losses_R.append(loss_R.item())

                # loss_R.backward()
                # opt_rew.step()

                # batch_log += f"loss_r: {torch.mean((r_pred_properties - r_properties)**2 ).item():.2f}\t{torch.mean((f_pred_properties - f_properties)**2 ).item():.2f}\n"
                # batch_log += f"properties: {torch.mean(r_properties, 0)[0].item():.2f}\t{torch.mean(r_properties, 0)[1].item():.2f}\t{torch.mean(r_properties, 0)[2].item():.2f}\t{torch.mean(f_properties, 0)[0].item():.2f}\t{torch.mean(f_properties, 0)[1].item():.2f}\t{torch.mean(f_properties, 0)[2].item():.2f}\n"


                print_and_save(batch_log, f"{log_dir}/batch-log.txt")
        epoch_log += f"loss_d:{np.mean(losses_D):.4f}\t loss_g:{np.mean(losses_G):.4f}\t"
                        #loss_e:{np.mean(losses_E):.4f}\t loss_r:{np.mean(losses_R):.4f}\t
        print_and_save(epoch_log, f"{log_dir}/epoch-log.txt")

        # Save model checkpoints.
        if (epoch+1) % save_step == 0:
            for model in [encoder, generator, discriminator]: #, energy_model, reward_model
                path = os.path.join(models_dir, f'{ligand_size}-{epoch+1}-{str(type(model)).split(".")[-1][:-2]}.ckpt')
                torch.save(model.state_dict(), path)
            print(f'Saved model checkpoints into {models_dir}...')


if __name__ == '__main__':
    main()