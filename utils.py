from rdkit import Chem
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd
import math
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import pickle
import gzip

import warnings
warnings.filterwarnings("ignore")

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

atom_decoder = {0: 0, 1: 6, 2: 7, 3: 8, 4: 9, 5: 16, 6:17}
bond_decoder = {0: Chem.rdchem.BondType.ZERO,
                1: Chem.rdchem.BondType.SINGLE,
                2: Chem.rdchem.BondType.DOUBLE,
                3: Chem.rdchem.BondType.TRIPLE,
                4: Chem.rdchem.BondType.AROMATIC
                }

SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open('data/SA_score.pkl.gz')) for j in range(1, len(i))}

class MolecularMetrics(object):

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            mols])))

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                 spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in mols]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores


def print_and_save(s, fname):
    print(s)
    with open(fname,"a") as f:
        f.write(s + "\n")


def matrices2mol(node_labels, edge_labels):
    mol = Chem.RWMol()

    # Keep only non-zero nodes and edges.
    idx = np.nonzero(node_labels)[0]
    for node_label in node_labels[idx]:
        mol.AddAtom(Chem.Atom(atom_decoder[node_label]))
    edge_labels = edge_labels[idx][:, idx]
    for start, end in zip(*np.nonzero(edge_labels)):
        if start < end:
            mol.AddBond(int(start), int(end), bond_decoder[edge_labels[start, end]])
    try:
        Chem.SanitizeMol(mol)
    except:
        mol = None

    return mol


def compute_gradient_penalty(discriminator, r_atoms, r_bonds, f_atoms, f_bonds, mol_d):
    """Calculates the gradient penalty (L2_norm(dy/dx) - 1)**2"""
    # Random weight term for interpolation between real and fake samples
    alpha_atoms = Tensor(np.random.random((r_atoms.size(0), 1, 1)))
    alpha_bonds = alpha_atoms.unsqueeze(-1)
    # Get random interpolation between real and fake samples
    interp_atoms = (alpha_atoms * r_atoms + (1 - alpha_atoms) * f_atoms).requires_grad_(True)
    interp_bonds = (alpha_bonds * r_bonds + (1 - alpha_bonds) * f_bonds).requires_grad_(True)

    interp_atoms = F.gumbel_softmax(interp_atoms, tau=1, hard=True)
    interp_bonds = F.gumbel_softmax(interp_bonds, tau=1, hard=True)

    if mol_d:
        interp_validity = discriminator(interp_bonds, None, interp_atoms)
    else:
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


rdkit = MolecularMetrics()
def reward(mols):
    """Calaulate property scores of QED, logP, and SAS."""
    # validity = rdkit.valid_scores(mols)
    logp = rdkit.water_octanol_partition_coefficient_scores(mols, norm=True)
    sas = rdkit.synthetic_accessibility_score_scores(mols, norm=True)
    qed = rdkit.quantitative_estimation_druglikeness_scores(mols, norm=True)

    properties = np.stack((logp, sas, qed), 1)
    return properties

def compute_rdkit_property(r_atoms, r_bonds, f_atoms, f_bonds):
    # Retrieve non-one-hot embedding atoms and bonds.
    r_edges, r_nodes = torch.max(r_bonds, -1)[1], torch.max(r_atoms, -1)[1]
    f_edges, f_nodes = torch.max(f_bonds, -1)[1], torch.max(f_atoms, -1)[1]

    # Round adjacency matrix to be symmetric.
    f_edges = torch.round((f_edges + f_edges.permute(0, 2, 1))/2).to(torch.int32)

    r_mols = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy())
                                        for n_, e_ in zip(r_nodes, r_edges)]
    f_mols = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy())
                                        for n_, e_ in zip(f_nodes, f_edges)]

    r_properties = torch.from_numpy(reward(r_mols)).type(Tensor)
    f_properties = torch.from_numpy(reward(f_mols)).type(Tensor)

    return r_properties, f_properties

