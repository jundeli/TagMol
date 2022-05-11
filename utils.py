from rdkit import Chem
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

ligand_size = 36

class receptorNet(nn.Module):
    """Receptor encoding network."""
    def __init__(self, in_channels, out_channels, ksize=3):
        super(receptorNet, self).__init__()

        self.c1 = nn.Conv3d(in_channels, out_channels, ksize, padding=ksize//2)
        self.c1 = nn.utils.spectral_norm(self.c1)
        self.c2 = nn.Conv3d(out_channels, out_channels, ksize, padding=ksize//2)
        self.c2 = nn.utils.spectral_norm(self.c2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = self.bn1(x)
        h = nn.ReLU()(residual)
        _, _, ht, wt = h.size() # for interpolation if necessary
        h = self.c1(h)
        h = self.bn2(h)
        h = nn.ReLU()(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 4)
        out = h.view(h.size(0), -1)

        return out

class ligandNet(nn.Module):
    """Ligand generator network."""
    def __init__(self, conv_dims):
        super(ligandNet, self).__init__()
        self.num_atoms = len(ligAtom)
        self.num_bonds = len(bondType)

        layers = []
        for c0, c1 in zip([1000]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=0.5, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.atom_layer = nn.Linear(conv_dims[-1], ligand_size * self.num_atoms)
        self.bonds_layer = nn.Linear(conv_dims[-1], ligand_size * ligand_size * self.num_bonds)
        
    def forward(self, x):
        out = self.layers(x)
        atoms_logits = self.atom_layer(out).view(-1, ligand_size, self.num_atoms)
        atoms_logits = nn.Dropout(p=0.5)(atoms_logits)

        ### TODO: check whether to move bonds to 2nd dim
        bonds_logits = self.bonds_layer(out).view(-1, ligand_size, ligand_size, self.num_bonds)
        bonds_logits = (bonds_logits + bonds_logits.permute(0, 2, 1, 3)) / 2
        bonds_logits = nn.Dropout(p=0.5)(bonds_logits)

        return atoms_logits, bonds_logits

bondType = {0.0: np.int8(0), 1.0: np.int8(1), 2.0: np.int8(2), 3.0: np.int8(3), 1.5: np.int8(4)}
ligAtom = {None: 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'X': 6}

def print_and_save(s, fname):
    print(s)
    with open(fname,"a") as f:
        f.write(s + "\n")

def postprocess(inputs, method, temperature=1.):
    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]
    def delistify(x):
        return x if len(x) > 1 else x[0]

    if method == 'soft_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                    / temperature, hard=False).view(e_logits.size())
                    for e_logits in listify(inputs)]
    elif method == 'hard_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                    / temperature, hard=True).view(e_logits.size())
                    for e_logits in listify(inputs)]
    else:
        softmax = [F.softmax(e_logits / temperature, -1)
                    for e_logits in listify(inputs)]

    return [delistify(e) for e in (softmax)]

atom_decoder_m = {i: l for i, l in enumerate(ligAtom)}
def matrices2mol(self, node_labels, edge_labels, strict=False):
    mol = Chem.RWMol()

    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end:
            mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None

    return mol

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

# # Get pre-trained the NuralDock model.
# def compute_dock_energy(neuraldock, model, data_loader):
#     model.eval()
#     neuraldock.eval()
#     real_de, fake_de = 0, 0
#     for recs, atoms, bonds in tqdm(data_loader):
#         atoms_logits, bonds_logits = model(recs)
#         real_de += neuraldock(recs, atoms, bonds)[0].sum()
#         fake_de += neuraldock(recs, atoms_logits, bonds_logits)[0].sum()
    
#     model.train()
#     size = len(data_loader.dataset)
#     return real_de/size, fake_de/size