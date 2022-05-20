#######################################################
# This is an unofficial PyTorch implementation for 
# https://bitbucket.org/dokhlab/neuraldock/src/master/
#######################################################

import os, time
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
lr             = 1e-6
beta1          = 0.0
beta2          = 0.9
batch_size     = 16
max_epoch      = 3000
num_workers    = 2
ligand_size    = 36
save_step      = 200

name = "neural"
log_fname = f"{name}/logs"
viz_dir = f"{name}/viz"
models_dir = f"{name}/saved_models"

if not os.path.exists(log_fname):
    os.makedirs(log_fname)
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

class recEncoder(nn.Module):
    """Protein receptor encoding network."""
    def __init__(self):
        super(recEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.utils.spectral_norm(nn.Linear(8000, 1024)),
            nn.LeakyReLU())
        self.block2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 1024)),
            nn.LeakyReLU())

    def forward(self, x):
        # Encode receptor features.
        h = x.view(x.size(0), -1)
        h = self.block1(h)
        for _ in range(9):
            h += self.block2(h)
        
        return h

class ligEncoder(nn.Module):
    """Ligand encoding network."""
    def __init__(self):
        super(ligEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.utils.spectral_norm(nn.Linear(6732, 1024)),
            nn.LeakyReLU())
        self.block2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 1024)),
            nn.LeakyReLU())

    def forward(self, a, b):
        # Encode ligand features.
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)
        h = torch.cat((a, b), -1)
        h = self.block1(h)
        for _ in range(9):
            h += self.block2(h)
        
        return h

class NeuralDock(nn.Module):
    """Network for predicting docking energy."""
    def __init__(self):
        super(NeuralDock, self).__init__()
        self.rec_enc = recEncoder()
        self.lig_enc = ligEncoder()
        self.block1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.utils.spectral_norm(nn.Linear(2048, 1024)),
            nn.LeakyReLU())
        self.block2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 1024)),
            nn.LeakyReLU())
        self.energy_layer = nn.utils.spectral_norm(nn.Linear(1024, 1))
        self.stat_layer = nn.utils.spectral_norm(nn.Linear(1024, 13*7))

    def forward(self, recs, atoms, bonds):
        enc_recs = self.rec_enc(recs)
        enc_ligs = self.lig_enc(atoms, bonds)
        h = torch.cat((enc_recs, enc_ligs), -1)
        h = self.block1(h)
        for _ in range(9):
            h += self.block2(h)

        # Output binding energy and 13x7 summary statistics.
        bd = self.energy_layer(h)
        stat = self.stat_layer(h)

        return bd, stat

# Make the optimizer.
model = torch.nn.DataParallel(NeuralDock())
optimizer = torch.optim.Adam(model.parameters(), lr)

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
                        batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,
                        num_workers=num_workers)
                        
# checkpoint = torch.load(f"{models_dir}/neuraldock-2000.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# epoch_start = checkpoint['epoch']
epoch_start = 0

# train loop
print('Start traning...')
for epoch in tqdm(range(epoch_start, max_epoch),  desc='total progress'):
    model.train()
    losses = []
    for batch, (recs, atoms, bonds, bd, stats) in enumerate(train_loader):
        curr_log = f"epoch {epoch}\t"
        print()

        # Train the model.
        optimizer.zero_grad()
        bd_pred, stats_logits = model(recs, atoms, bonds)
        print(bd_pred.shape, stats_logits.shape)
        loss = torch.nn.MSELoss(reduction='mean')(bd_pred, bd) + \
               torch.nn.MSELoss(reduction='mean')(stats_logits, stats)
        loss.backward()
        losses.append(loss.item())
        print(f"{epoch}:{batch}\t{loss.item():.4f}", end="\r")
        optimizer.step()

    curr_log += f"loss:{np.mean(losses):.4f}\t"
    print_and_save(curr_log, "pdb-cla.txt")

    if (epoch+1) % save_step == 0 or np.mean(losses) < 650:
        torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, f"{models_dir}/neuraldock-{epoch+1}.pth")
