from rdkit import Chem
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def print_and_save(s, fname):
    print(s)
    with open(fname,"a") as f:
        f.write(s + "\n")


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


def compute_gradient_penalty(discriminator, r_atoms, r_bonds, f_atoms, f_bonds):
    """Calculates the gradient penalty (L2_norm(dy/dx) - 1)**2"""
    # Random weight term for interpolation between real and fake samples
    alpha_atoms = Tensor(np.random.random((r_atoms.size(0), 1, 1)))
    alpha_bonds = alpha_atoms.unsqueeze(-1)
    # Get random interpolation between real and fake samples
    interp_atoms = (alpha_atoms * r_atoms + (1 - alpha_atoms) * f_atoms).requires_grad_(True)
    interp_bonds = (alpha_bonds * r_bonds + (1 - alpha_bonds) * f_bonds).requires_grad_(True)

    interp_atoms = F.gumbel_softmax(interp_atoms, tau=1, hard=True)
    interp_bonds = F.gumbel_softmax(interp_bonds, tau=1, hard=True)

    interp_validity = discriminator((interp_atoms, interp_bonds))
    fake = Variable(Tensor(r_atoms.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=interp_validity,
        inputs=(interp_atoms, interp_bonds),
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True # adj gradients not used in GATLayer
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

    