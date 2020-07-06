# Funciton added: save the crystal info and reload
import torch
from torch.utils.data import Dataset,DataLoader
import os
import functools
import numpy as np
from pymatgen.core.structure import Structure
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random

class CrystalDataset(Dataset):
    '''
    Crystal structure dataset
    Args:
    id_prop_file (string): a dict that stores id and properties, a torch.save binary file. 
    root_dir (string): Directory of the crystal structures
    transform (callable, optional): Optional transform to be applied on a sample.
    In this case, we'll transform our crystals into tensors
    Returns:
    sample: dict, {'crystal_id':crystal_id,'crystal_fea':crystal_tsfm,'property':cry_prop}
    Crystal features after transformation and corresponding property.
    If site == True, then property is an 1d numpy array; otherwise it is a real number. 
    '''
    def __init__(self,id_prop_file,root_dir,transform=None,shuffle=True,preprocessed=False):
        self.root_dir = root_dir
        self.transform= transform
        self.crystal_info = torch.load(id_prop_file)
        self.preprocessed=preprocessed
        self.ids = list(self.crystal_info.keys())
        if shuffle:
            random.shuffle(self.ids)
    
    def __len__(self):
        return len(self.crystal_info)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self,idx):
        '''
        idx: integer 
        '''
        cry_id = self.ids[idx]
        cry_prop = self.crystal_info[cry_id]
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
        self.ele_fea = torch.load(atom_fea_file)
        self.root_dir = root_dir
        self.ncut = ncut
        self.rcut = rcut
        self.fmat = fmat

    def __call__(self,sample,preprocessed=False):
        #preprocessed: bool, whether to read in crystal features from crystal_id.save files.
        crystal_id,cry_prop = sample['crystal_id'],sample['property']
        crystal_tsfmed  = self.crystal_embed(crystal_id,preprocessed)
        return {'crystal_id':crystal_id,'crystal_fea':crystal_tsfmed,'property':cry_prop}

    def get_atom_fea(self,ele_symbols):
        a_fea = []
        for ele in ele_symbols:
            a_fea.append(self.ele_fea[ele])
        return np.array(a_fea)

    def crystal_embed(self,c_id,preprocessed):
        if preprocessed:
            atom_fea,nbr_fea,nbr_idx = torch.load(os.path.join(self.root_dir,c_id+self.fmat))
            return atom_fea,nbr_fea,nbr_idx
        else:
            crystal = Structure.from_file(os.path.join(self.root_dir,c_id+self.fmat))
            elements = crystal.species
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
            atom_fea = torch.tensor(self.get_atom_fea([i.name for i in elements])).float()
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
        n = atom_fea.shape[0]  # number of atoms in this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_idx.append(nbr_idx+base_idx)
        crystal_idx.append(torch.tensor(np.arange(n)+base_idx))
        base_idx = base_idx + n
    return (crystal_idx,torch.cat(batch_atom_fea,dim=0),torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_idx, dim=0),torch.tensor(batch_prop).reshape(-1,1).float())

def collect_pool_site(dataset_list):
    '''
   Collect batch features and properties for site properties. 
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
    batch_prop = np.concatenate(batch_prop)
    return (crystal_idx,torch.cat(batch_atom_fea,dim=0),torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_idx, dim=0),torch.tensor(batch_prop).reshape(-1,1).float())

def save_preprocess():
    # save preprocessed data
    tsfm = Crystal_embeding('ele_rep.save',root_dir='../icsdata/')
    crydata = CrystalDataset(csv_file='id_prop.csv',root_dir='../icsdata/',transform=tsfm,preprocessed=False)
    n = len(crydata)
    for i in range(n):
        a = crydata[i]
        b = a['crystal_fea']
        torch.save((b[0],b[1],b[2]),os.path.join('../icsdata/',a['crystal_id']+'.save')) 
