#######################################################
# This is a PyTorch implementation for deterministic
# Rec2Lig regression model.
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

name = "rec2lig"
log_fname = f"{name}/logs/pdb"
viz_dir = f"{name}/viz/pdb"
models_dir = f"{name}/saved_models/pdb"

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

class Receptor2Ligand(nn.Module):
    """Network for mapping receptors to ligands."""
    def __init__(self, conv_dims, in_channels=8, out_channels=3, activation=nn.ReLU()):
        super(Receptor2Ligand, self).__init__()
        self.num_atoms = len(ligAtom)
        self.num_bonds = len(bondType)

        self.activation = activation
        self.ch = gen_dim
        self.block1 = BasicBlock(in_channels, self.ch, activation=activation)
        self.block2 = BasicBlock(self.ch, self.ch, activation=activation)
        self.block3 = BasicBlock(self.ch, out_channels, activation=activation)
        self.bn = nn.BatchNorm2d(in_channels)

        layers = []
        for c0, c1 in zip([3000]+conv_dims[:-1], conv_dims):
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

    def forward(self, x):
        # Encode receptor features.
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        # h = F.avg_pool3d(h, 2)
        h = h.view(h.size(0), -1)

        # Generate atoms and bonds.
        out = self.layers(h)
        atoms_logits = self.atom_layer(out).view(-1, ligand_size, self.num_atoms)
        atoms_logits = nn.Softmax(dim=-1)(atoms_logits)

        ### TODO: check whether to move bonds to 2nd dim.
        bonds_logits = self.bonds_layer(out).view(-1, ligand_size, ligand_size, self.num_bonds)
        bonds_logits = (bonds_logits + bonds_logits.permute(0, 2, 1, 3)) / 2.0
        bonds_logits = nn.Softmax(dim=-1)(bonds_logits)

        return atoms_logits, bonds_logits

# Make the optimizer.
model = torch.nn.DataParallel(Receptor2Ligand(conv_dims))
optimizer = torch.optim.Adam(model.parameters(), lr, (beta1, beta2))
if resume_step:
    checkpoint = torch.load(f"{models_dir}"+"/baseline.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = resume_step
else:
    start_epoch = 0


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

# Load NeuralDock for validation.
# neuraldock = NeuralDock()

# TODO: Get the trained NuralDock model.
def compute_dock_energy(neuraldock, model, data_loader):
    neuraldock.eval()
    real_de, fake_de = 0, 0
    for recs, atoms, bonds in tqdm(data_loader):
        atoms_logits, bonds_logits = model(recs)
        real_de += neuraldock(recs, atoms, bonds).sum()
        fake_de += neuraldock(recs, atoms_logits, bonds_logits).sum()
    
    size = len(data_loader.dataset)
    return real_de/size, fake_de/size

def main():
    # train loop
    print('Start traning...')
    lowest_loss = np.inf
    for epoch in tqdm(range(start_epoch, max_epoch),  desc='total progress'):
        model.train()
        losses = []
        for batch, (recs, atoms, bonds, bd, medusa) in enumerate(train_loader):
            curr_log = f"epoch {epoch+1}\t"

            # Train the model.
            optimizer.zero_grad()
            atoms_logits, bonds_logits = model(recs)
            loss = torch.nn.MSELoss(reduction='sum')(atoms_logits, atoms) + \
                torch.nn.MSELoss(reduction='sum')(bonds_logits, bonds)
            loss = loss / atoms.size(0)
            
            loss.backward()
            losses.append(loss.item())
            print(f"{epoch+1}:{batch}\t{loss.item():.4f}", end="\r")
            optimizer.step()

        curr_log += f"loss:{np.mean(losses):.4f}\t"
        print_and_save(curr_log, f"{log_fname}/log.txt")

        # # TODO: Varify atom labels before visualization.
        # if visualization:
        #     (atoms_hard, bonds_hard) = postprocess((atoms_logits, bonds_logits), 'hard_gumbel')
        #     atoms_hard, bonds_hard = torch.max(atoms_hard, -1)[1], torch.max(bonds_hard, -1)[1]
        #     mols = [matrices2mol(a.item(), b.item(), strict=True) for a, b in zip(atoms_hard, bonds_hard)]

        if (epoch+1) % save_step == 0:
            if np.mean(losses) < lowest_loss:
                lowest_loss = np.mean(losses)
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, f"{models_dir}/rec2lig-{epoch+1}.pth")

if __name__ == '__main__':
    main()