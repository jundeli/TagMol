import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromMol2File
from rdkit.Chem.rdmolops import GetAdjacencyMatrix


class Normalize(object):
    """Normalize the protein atom points."""
    
    def __call__(self, sample):
        protein, ligand = sample['protein'], sample['ligand']

        n_nonzero = np.count_nonzero(protein[:, 3])
        protein[:n_nonzero, :3] = protein[:n_nonzero, :3] - np.expand_dims(np.mean(protein[:n_nonzero, :3], axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(protein[:n_nonzero, :3] ** 2, axis = 1)),0)
        protein[:n_nonzero, :3] = protein[:n_nonzero, :3] / dist #scale

        return {'protein': protein, 'ligand': ligand}


class RandomRotateJitter(object):
    """Apply random rotation and jitter for data augmentation."""

    def __call__(self, sample):
        protein, ligand = sample['protein'], sample['ligand']

        theta = np.random.uniform(0,np.pi*2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        protein[:,[0,2]] = protein[:,[0,2]].dot(rotation_matrix) # random rotation along axis=1

        n_nonzero = np.count_nonzero(protein[:, 3])
        protein[:n_nonzero, :3] += np.random.normal(0, 0.02, size=(n_nonzero, 3)) # random jitter

        return {'protein': protein, 'ligand': ligand}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        protein, (atoms, bonds) = sample['protein'], sample['ligand']
        
        return {'protein': torch.from_numpy(protein),
                'ligand': (torch.from_numpy(atoms), torch.from_numpy(bonds))}

                
class PDBbindPLDataset(Dataset):
    """PDBbind v2020 protein-ligand dataset."""

    def __init__(self, root_dir, n_points=8000, lig_size=32, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the protein-ligand complexes.
            n_points (int): Number of protein atoms to be extracted.
            lig_size (int): Maximum number of atoms to be kept in ligands.
            train (boolean): Splitting for train or test set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.n_points = n_points
        self.lig_size = lig_size
        self.split = 'train' if train else 'test'
        self.pids = self._read_pids()
        # All p-l complexes with heavy atoms orther than below are filtered out.
        self.atom_encoder = {None: 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S':5, 'Cl': 6}
        self.bond_encoder = {0.:0, 1.:1, 2.:2, 3.:3, 1.5:4}
        self.transform = transform

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        with open(os.path.join(self.root_dir, f'{pid}/{pid}_protein.pdb')) as f:
            # Read 3D coordinates of protein atoms in angstroms.
            protein = [[float(i) for i in (line[29:38], line[38:46], line[46:54])]+[self.atom_encoder[line.split()[2][0]]] \
                            for line in f.read().split('\n') if line[:4]=='ATOM' and line.split()[2][0]!='H' and not line.split()[2][0].isdigit()]
        # Load ligand from .mol2 file.
        ligand = MolFromMol2File(os.path.join(self.root_dir, f'{pid}/{pid}_ligand.mol2'))

        # Calculate ligand centroid and rank protein points by distance.
        l_centroid = np.mean(ligand.GetConformer().GetPositions(), 0)
        protein = np.asarray(sorted(protein, key=lambda x: np.linalg.norm(x[:-1] - l_centroid)))
        
        # Pad null points or select self.n_points atoms closest to l_centriod.
        if protein.shape[0] > self.n_points:
            protein = protein[:self.n_points]
        else:
            pad = np.repeat([[0, 0, 0, 0]], self.n_points - protein.shape[0], axis=0)
            protein = np.concatenate((protein, pad), 0)

        atoms = [self.atom_encoder[i.GetSymbol()] for i in list(ligand.GetAtoms())]
        bonds = GetAdjacencyMatrix(ligand, useBO=True)
        if len(atoms) > self.lig_size:
            indices = np.argpartition(np.sum(bonds, axis=0), len(atoms)-self.lig_size) # rank atoms by bond orders
            indices = sorted(indices[:len(atoms)-self.lig_size], reverse=True) # indices to be removed

            for i in indices:
                bonds = np.concatenate((bonds[:i,:], bonds[i+1:,:]), axis=0)
                bonds = np.concatenate((bonds[:,:i], bonds[:,i+1:]), axis=1)
                atoms = np.concatenate((atoms[:i], atoms[i+1:]))
        else:
            bonds = np.pad(bonds, (0, self.lig_size-len(atoms)))
            atoms = np.pad(atoms, (0, self.lig_size-len(atoms)))
        
        # One-hot encoding for atoms and bonds
        atoms = np.eye(len(self.atom_encoder))[atoms.astype(int)]
        bonds = np.vectorize(self.bond_encoder.get)(bonds)
        bonds = np.eye(len(self.bond_encoder))[bonds]

        sample = {'protein': protein, 'ligand': (atoms, bonds)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _read_pids(self):
        with open(os.path.join(self.root_dir, f'index/{self.split}.txt'), 'r') as f:
            pids = f.read().splitlines()
        return pids

# Driver code for testing.
if __name__ == '__main__':
    # import warnings
    # warnings.filterwarnings("ignore")

    train_dataset = PDBbindPLDataset(root_dir='data/pdbbind/refined-set',
                                        n_points=5000, 
                                        lig_size=36,
                                        train=True,
                                        transform=transforms.Compose([
                                            Normalize(),
                                            RandomRotateJitter(),
                                            ToTensor()
                                        ]))

    train_dataloader = DataLoader(train_dataset, batch_size=16,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched['protein'].size(),
            sample_batched['ligand'][0].size(),
            sample_batched['ligand'][1].size())
        print(sample_batched['protein'][0])
        print(sample_batched['ligand'][0][0])
        print(sample_batched['ligand'][1][0])
        break
