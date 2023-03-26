import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloader import PDBbindPLDataset, Normalize, RandomRotateJitter, ToTensor

n_pc_points    = 4096
ligand_size    = 14
dataset        = 'refined'


train_dataset = PDBbindPLDataset(root_dir=f'data/pdbbind/{dataset}-set',
                                n_points=n_pc_points,
                                lig_size=ligand_size,
                                train=True,
                                transform=transforms.Compose([
                                    Normalize(),
                                    RandomRotateJitter(sigma=0.0),
                                    ToTensor()
                                ]))
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset),
                        shuffle=True, drop_last=False)    
                        
for batch, sample_batched in enumerate(train_loader):
    proteins = sample_batched['protein']
    r_atoms, r_bonds = sample_batched['ligand']


pickle.dump(proteins, open(f'data/pdbbind/{dataset}-set/proteins.p', "wb"))
pickle.dump(r_atoms, open(f'data/pdbbind/{dataset}-set/atoms.p', "wb"))
pickle.dump(r_bonds, open(f'data/pdbbind/{dataset}-set/bonds.p', "wb"))

