# Funciton added: save the crystal info and reload
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import functools
import numpy as np
from pymatgen.core.structure import Structure
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class CrystalDataset(Dataset):
    '''
    Crystal structure dataset
    Args:
    csv_file (string): Path to the csv file with annotations and properties.
    In our case, the first column is the file id (crystal id), the second column is the property.
    root_dir (string): Directory with all the crystals
    transform (callable, optional): Optional transform to be applied on a sample.
    In this case, we'll transform our crystals into tensors
    Returns:
    sample: dict, {'crystal_id':crystal_id,'crystal_fea':crystal_tsfm,'property':cry_prop}
    Crystal features after transformation and corresponding property.
    '''
    def __init__(self,csv_file,root_dir,transform=None,shuffle=True,preprocessed=False):
        self.root_dir = root_dir
        self.transform= transform
        crystal_info = pd.read_csv(csv_file,header=None)
        self.preprocessed=preprocessed
        if shuffle:
            self.crystal_info = crystal_info.sample(n=crystal_info.shape[0])
        else:
            self.crystal_info = crystal_info

    def __len__(self):
        return len(self.crystal_info)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self,idx):
        '''
        idx: integer 
        '''
        cry_id = self.crystal_info.iloc[idx,0]
        cry_prop = self.crystal_info.iloc[idx,1] # only 1 property
        sample = {'crystal_id':cry_id,'property':cry_prop}
        if self.transform:
            sample = self.transform(sample,self.preprocessed)
        return sample

class Crystal_embeding(object):
    '''
    Transform the crystal structure into tensor representations
    Inputs:
    atom_fea_file: a csv file that contains the atomic features for 
    each element.
    fmat: format of our input crystal structures
    ncut: maximum number of neighbors
    rcut: distance cutoff. Try increase rcut if you cannot found enough
    neighbors.
    '''
    def __init__(self,atom_fea_file,root_dir,fmat='.cif',ncut=12,rcut=15):
        ele_fea = pd.read_csv(atom_fea_file,header=None)
        self.ele_fea = ele_fea.set_index(0).T.to_dict('list')
        self.root_dir = root_dir
        self.ncut = ncut
        self.rcut = rcut
        self.fmat = fmat

    def __call__(self,sample,preprocessed=False):
        #preprocessed: bool, whether to read in crystal features from crystal_id.save files. 
        crystal_id,cry_prop = sample['crystal_id'],sample['property']
#        if preprocessed:
#            crystal_tsfmed = self.read_preprocessed(crystal_id)
#        else:
        crystal_tsfmed  = self.crystal_embed(crystal_id,preprocessed)
        return {'crystal_id':crystal_id,'crystal_fea':crystal_tsfmed,'property':cry_prop}

    def read_preprocessed(self,c_id):
        crystal = Structure.from_file(os.path.join(self.root_dir,c_id+self.fmat))
        elements = crystal.species
        n = crystal.num_sites # number of atoms in that crystal
        atom_fea = self.get_atom_fea([i.name for i in elements])
        atom_fea,nbr_fea,nbr_idx = torch.load(c_id+'.save')
        return atom_fea,nbr_fea,nbr_idx
    
    def get_atom_fea(self,ele_symbols):
        a_fea = []
        for ele in ele_symbols:
            a_fea.append(self.ele_fea[ele])
        return np.array(a_fea)

    def crystal_embed(self,c_id,preprocessed):
        crystal = Structure.from_file(os.path.join(self.root_dir,c_id+self.fmat))
        elements = crystal.species
        n = crystal.num_sites # number of atoms in that crystal
        atom_fea = self.get_atom_fea([i.name for i in elements])
        atom_fea = torch.tensor(atom_fea).float()
        # the neighbor info won't change, thus we can save the preprocessed neighboring
        # features and load them easily, this will save a lot of computational time. 
        if preprocessed:
            nbr_fea,nbr_idx = torch.load(os.path.join(self.root_dir,c_id+'.save'))
        else:
            all_nbrs = crystal.get_all_neighbors(self.rcut, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_idx = []
            nbr_fea = []
            for nbr in all_nbrs:
                if len(nbr)<self.ncut:
                    print(c_id)
                nbr_idx.append(list(map(lambda x: x[2],nbr[:self.ncut])))
                nbr_fea.append(list(map(lambda x: x[1],nbr[:self.ncut])))
            nbr_fea,nbr_idx = np.array(nbr_fea),torch.tensor(np.array(nbr_idx))
            nbr_fea = torch.tensor(self.gaussian_expand_dis(nbr_fea)).float()
        return atom_fea,nbr_fea,nbr_idx
      
    def gaussian_expand_dis(self,dis,l=5,nbin=50,sigma=0.5):
        m,n = dis.shape
        r0 = np.linspace(0,l,nbin,False)
        dis = dis.reshape(-1,1)
        expo = (dis - r0)**2/sigma**2
        return np.exp(-expo).reshape(m,n,nbin)

def collect_pool(dataset_list):
    '''
    The crystals have different number of atoms, thus we need to stack 
    the crystals to get a batch.
    The atom indices are simply shifted along with the crystal id. 
    '''
    batch_atom_fea, batch_nbr_fea, batch_nbr_idx = [], [], []
    crystal_idx, batch_prop = [], []
    base_idx = 0
    for sample in dataset_list:
        prop = sample['property']
        batch_prop.append(prop)
        atom_fea,nbr_fea,nbr_idx = sample['crystal_fea']
        n = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_idx.append(nbr_idx+base_idx)
        crystal_idx.append(torch.tensor(np.arange(n)+base_idx))
        base_idx = base_idx + n
    return (crystal_idx,torch.cat(batch_atom_fea,dim=0),torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_idx, dim=0),torch.tensor(batch_prop).reshape(-1,1).float())

#tsfm = Crystal_embeding('element_fea.csv',root_dir='./icsdata/')
#crydata = CrystalDataset(csv_file='id_prop.csv',root_dir='./icsdata/',transform=tsfm,preprocessed=True)

#n = len(crydata)
# save preprocessed data
#for i in range(n):
#    a = crydata[i]
#    b = a['crystal_fea']
#    torch.save((b[1],b[2]),a['crystal_id']+'.save') 
