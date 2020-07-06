import numpy as np
from pylab import plot,show
import torch
import pandas as pd

class LoadFile(object):
    '''
    input: LAMMPS dump file or an already preprocessed binary file
    fname: string, the file to be read
    n_attribute: number of attributes for each atom. For example, if you dump
    type, id, x, y, z for each atom, then n_attribute should be 5. 
    output: Sorted positions Sr, a nframe*ntot*3 array; lattice parameter,
    number of atoms
    '''
    def __init__(self,fname,n_attribute):
        self.fname = fname
        self.nattr = n_attribute # number of attributes for each atom

    def loadpos(self): # load information from a LAMMPS dump file
        with open(self.fname,'r') as f:
            head = [next(f) for x in range(4)]
            ntot = int(head[3])
            atom_attr = []
            lattice = []
            for line in f:
                try:
                    li = line.split()
                    if len(li) == 2:
                        [lattice.append(float(x)) for x in li]
                    elif len(li) == self.nattr:
                        [atom_attr.append(float(x)) for x in li]
                except:
                    line.startswith('I')
        lattice = np.array(lattice).reshape(-1,2) # 3*nframe*2 array
        lattice = lattice[:,1] - lattice[:,0]
        self.ntot = ntot
        self.nframe = lattice[::3].shape[0]
        self.lc = lattice.reshape(self.nframe,3)
        atom_attr = np.array(atom_attr).reshape(-1,self.nattr)
        self.pos = atom_attr[:,2:].reshape(self.nframe,self.ntot,3)

def get_neighbor(pos,l,ncut=14):
    # 14 nearest neighbors for particle A
    n = pos.shape[0]
    neigh_id = [] # record the atom id
    dis = [] # record the distances
    for i in range(n):
        tmp = pos[i]-pos
        tmp = tmp - np.rint(tmp/l)*l
        tmp = np.linalg.norm(tmp,axis=1)
        nn= tmp.argsort()[1:ncut]
        neigh_id.append(nn)
        dis.append(tmp[nn])
    return np.array(neigh_id),np.array(dis)

def expand_dis(dis,l=5,nbin=50,sigma=0.5):
    m,n = dis.shape
    r0 = np.linspace(0,l,nbin,False)
    dis = dis.reshape(-1,1)
    expo = (dis - r0)**2/sigma**2
    return np.exp(-expo).reshape(m,n,nbin)

l = LoadFile('dump.dat',5)
l.loadpos()

gmap = []
expdis = []
base_id = 0
n1 = 3200
ntot = 4000
atom_fea = torch.load('atom_fea.save')
for i in range(41):
    graph_map,dis = get_neighbor(l.pos[i],l.lc[i])
    dis = expand_dis(dis[:ntot])
    nbr_idx = torch.tensor(graph_map)
    nbr_fea = torch.tensor(dis).float()
    torch.save((atom_fea,nbr_fea,nbr_idx),str(i)+'.save')


