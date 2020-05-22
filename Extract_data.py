from pymatgen import Structure
import numpy as np
from sklearn.preprocessing import label_binarize
import torch
import pickle
#from pymatgen.io.vasp import Poscar
#p = Poscar.from_file('POSCAR-0')

data = []
label = []
eform = torch.tensor(np.loadtxt('eform.dat')).float()

def batchlize(ntot,batch_size,shuf=True):
    a = np.array(range(ntot))
    if shuf:
        np.random.shuffle(a)
    return np.array([a[i:i+batch_size] for i in range(0,ntot,batch_size)])

def expand_dis(dis,l=4,nbin=40,sigma=0.5):
    m,n = dis.shape
    r0 = np.linspace(0,l,nbin,False)
    dis = dis.reshape(-1,1)
    expo = (dis - r0)**2/sigma**2
    return np.exp(-expo).reshape(m,n,nbin)

def getdata(batch,fname='POSCAR-',ncut=16):
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_idx = []
    batch_crystal_id = []
    base_id = 0
    for i in batch:
        crystal = Structure.from_file(fname+str(i))
        a = crystal.species
        n = crystal.num_sites # number of atoms in that crystal
        atom_fea = label_binarize([i.name for i in a],classes=['Al','B'])
        all_nbrs = crystal.get_all_neighbors(4, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_idx = []
        nbr_fea = []
        for nbr in all_nbrs:
            nbr_idx.append(list(map(lambda x: x[2],nbr[:ncut])))
            nbr_fea.append(list(map(lambda x: x[1],nbr[:ncut])))
        nbr_fea,nbr_idx = np.array(nbr_fea),np.array(nbr_idx)
        batch_crystal_id.append(np.arange(n)+base_id)
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_idx.append(nbr_idx+base_id)
        base_id =  base_id + n
    batch_nbr_fea = torch.tensor(expand_dis(np.vstack(batch_nbr_fea)))
    batch_crystal_id = [torch.tensor(i) for i in batch_crystal_id]
    batch_atom_fea = torch.tensor(np.vstack(batch_atom_fea))
    batch_nbr_idx = torch.tensor(np.vstack(batch_nbr_idx))
    return batch_crystal_id,batch_atom_fea,batch_nbr_fea,batch_nbr_idx

batches = batchlize(2600,100)
nbatch = batches.shape[0]
eform = [eform[batch] for batch in batches]
for i in range(nbatch):
    batch_crystal_id,batch_atom_fea,batch_nbr_fea,batch_nbr_idx = getdata(batches[i])
    data.append((batch_crystal_id,batch_atom_fea.float(),batch_nbr_fea.float(),batch_nbr_idx))
    label.append(eform[i])

torch.save(data,'data.save')
torch.save(label,'label.save')

