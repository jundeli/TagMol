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
lr             = 1e-4
beta1          = 0.0
beta2          = 0.9
batch_size     = 16
max_epoch      = 400
num_workers    = 2
ligand_size    = 36
gen_dim        = 64
conv_dims      = [4096, 2048, 1024]
visualization  = False
save_step      = 100
latent_dim     = 2
num_samples    = 10
start_epoch    = 0

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
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c3 = nn.Conv3d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c1 = nn.utils.spectral_norm(self.c1)
        self.c2 = nn.utils.spectral_norm(self.c2)
        self.c3 = nn.utils.spectral_norm(self.c3)

    def forward(self, x):
        h = self.c1(x)
        h = self.activation(h)
        h = self.c2(h)
        h = self.activation(h)
        h = self.c3(h)
        out = self.activation(h)
        return out

class RecPredictor(nn.Module):
    """Deterministic network for mapping receptors to ligands."""
    def __init__(self, in_channels, out_channels=3, hidden_channels=None, activation=nn.ReLU()):
        super(RecPredictor, self).__init__()

        self.activation = activation
        self.ch = gen_dim
        self.latent_dim = latent_dim
        self.block1 = BasicBlock(in_channels, self.ch, hidden_channels, activation=activation)
        self.block2 = BasicBlock(self.ch, self.ch, hidden_channels, activation=activation)
        self.block3 = BasicBlock(self.ch, out_channels, hidden_channels, activation=activation)

        self.fc_mu = nn.Linear(3000, self.latent_dim)
        self.fc_var = nn.Linear(3000, self.latent_dim)

    def forward(self, x):
        # Encode receptor features.
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        out = h.view(h.size(0), -1)

        mu = self.fc_mu(out)
        log_var = self.fc_var(out)

        return out, mu, log_var

class LV_LigNet(nn.Module):
    """Deep probabilistic regression for mapping receptors to ligand distributions."""
    def __init__(self, conv_dims):
        super(LV_LigNet, self).__init__()
        self.num_atoms = len(ligAtom)
        self.num_bonds = len(bondType)
        self.latent_dim = latent_dim
        self.rec_predictor = torch.nn.DataParallel(RecPredictor(8, out_channels=3, hidden_channels=5))

        layers = []
        for c0, c1 in zip([3000+self.latent_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
        self.layers = nn.Sequential(*layers)

        self.atom_layer = nn.Sequential(
                          nn.Linear(conv_dims[-1], 2048),
                          nn.ReLU(),
                          nn.Linear(2048, ligand_size * self.num_atoms),
                          nn.Dropout(p=0.5)
                          )
        self.bonds_layer = nn.Sequential(
                          nn.Linear(conv_dims[-1], 2048),
                          nn.ReLU(),
                          nn.Linear(2048, ligand_size * ligand_size * self.num_bonds),
                          nn.Dropout(p=0.5)
                          )

    def gmm_sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        samples = mu + std * torch.randn(num_samples, latent_dim) # shape = (num_samples, latent_dim)
        samples = samples.unsqueeze(0).repeat(batch_size, 1, 1) # shape = (N, num_samples, latent_dim)
        return samples

    def forward(self, rec_enc, mu, log_var):

        # Generate atoms and bonds.
        z = self.gmm_sampling(mu, log_var)
        rec_enc = rec_enc.unsqueeze(1).repeat(1, num_samples, 1) # shape = (N, num_samples, 3000)
        h = torch.cat((rec_enc, z), -1) # shape = (N, num_samples, 3000+latent_dim)
        out = self.layers(h)
        atoms_logits = self.atom_layer(out).view(out.size(0), -1, ligand_size, self.num_atoms)
        atoms_logits = nn.Softmax(dim=-1)(atoms_logits)

        ### TODO: check whether to move bonds to 2nd dim.
        bonds_logits = self.bonds_layer(out).view(out.size(0), -1, ligand_size, ligand_size, self.num_bonds)
        bonds_logits = (bonds_logits + bonds_logits.permute(0, 1, 3, 2, 4)) / 2.0
        bonds_logits = nn.Softmax(dim=-1)(bonds_logits)

        return atoms_logits, bonds_logits

# Make the optimizer.
model = torch.nn.DataParallel(LV_LigNet(conv_dims))
optimizer = torch.optim.Adam(model.parameters()), lr, (beta1, beta2)

# Make the dataloaders.
(trainingData, medusa, training) = pickle.load(open('data/tutorialData.pkl', 'rb'))
receptor, bonds, atoms, bd = zip(*[trainingData[pdbid] for pdbid in training])
receptor = torch.tensor(np.concatenate(receptor)).permute((0, 4, 1, 2, 3)).float()
atoms, bonds = torch.tensor(atoms).float(), torch.tensor(bonds).float()
bd = torch.tensor(bd).float()
medusa = torch.tensor(np.concatenate([medusa[pdbid] for pdbid in training])).float()

train_loader = torch.utils.data.DataLoader(list(zip(receptor, atoms, bonds, bd, medusa)), 
                        batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,
                        num_workers=num_workers)

test = pickle.load(open('data/coreData.pkl','rb'))
receptor, bonds, atoms, bd = zip(*[test[pdbid] for pdbid in list(test.keys())])
receptor = torch.tensor(np.concatenate(receptor)).permute((0, 4, 1, 2, 3)).float()
atoms, bonds = torch.tensor(atoms).float(), torch.tensor(bonds).float()
bd = torch.tensor(bd).float()

test_loader = torch.utils.data.DataLoader(list(zip(receptor, atoms, bonds, bd)),
                        batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True,
                        num_workers=num_workers)

def main():
    # train loop
    print('Start traning...')
    for epoch in tqdm(range(start_epoch, max_epoch),  desc='total progress'):
        model.train()
        losses = []
        for batch, (recs, atoms, bonds, bd, medusa) in enumerate(train_loader):
            curr_log = f"epoch {epoch+1}\t"

            # Train the model.
            optimizer.zero_grad()
            lig_pred, mu, logvar = model.rec_predictor(recs)
            atoms_logits, bonds_logits = model(lig_pred, mu, logvar)
            
            atoms = atoms.unsqueeze(1).repeat(1, num_samples, 1, 1)
            bonds = bonds.unsqueeze(1).repeat(1, num_samples, 1, 1, 1)

            atom_loss = torch.mean(torch.sum(torch.square(atoms_logits-atoms), (-2, -1)))
            bond_loss = torch.mean(torch.sum(torch.square(bonds_logits-bonds), (-3, -2, -1)))
            batch_loss = atom_loss + bond_loss
            
            batch_loss.backward()
            losses.append(batch_loss.item())
            print(f"{epoch+1}:{batch}\t{batch_loss.item():.4f}", end="\r")
            optimizer.step()

        curr_log += f"loss:{np.mean(losses):.4f}\t"
        print_and_save(curr_log, f"{log_fname}/log.txt")

if __name__ == '__main__':
    main()