# Energy-based Generative Models for Target-specific Drug Discovery
Pytorch implementation of Target-specific Generation of Molecules (TagMol). This library refers to the following source code.
* [yongqyu/MolGAN-pytorch](https://github.com/yongqyu/MolGAN-pytorch)
* [jundeli/quantum-gan]([https://github.com/jundeli/quantum-gan]


For details see [Energy-based Generative Models for Target-specific Drug Discovery](https://arxiv.org/pdf/2212.02404.pdf) by Junde Li, Collin Beaudoin, and Swaroop Ghosh.


## Dependencies

* **python>=3.5**
* **pytorch>=0.4.1**: https://pytorch.org
* **frechetdist**

## Structure
* [data](https://github.com/jundeli/TagMol/tree/main/data): should contain your protein-ligand complex datasets. Download PDBbind dataset from: https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/jul1512_psu_edu/ErM3Iuz_OjNMnHTsZyWVhGQBtLCakhUin4bqMShQWXEpKA?e=nRxKQy
* [models](https://github.com/jundeli/TagMol/blob/main/model.py): Class for Models.

## Training
```
python main.py
```
This main file was used for running TagMol experiments with GCN and GAT backends. The file data_dump.py could be executed first in order to run experiments faster with a preprocessed dataset.


Below are some generated molecules:

<div style="color:#0000FF" align="center">
<img src="molecules/mol1.png" width="430"/> 
<img src="molecules/mol2.png" width="430"/>
</div>

## Citation
```
@article{li2022energy,
  title={Energy-based Generative Models for Target-specific Drug Discovery},
  author={Li, Junde and Beaudoin, Collin and Ghosh, Swaroop},
  journal={arXiv preprint arXiv:2212.02404},
  year={2022}
}

```
