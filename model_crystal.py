#Handles crystal property predictions. Per crystal quantities like formation energy, band gap, bulk module etc.
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from pylab import plot,show,xlabel,ylabel,legend
import time

class conv_layer(torch.nn.Module):
    # torch.nn.ModuleList([conv_layer(1,2,3) for _ in range(3)]) creates 3 different convolution layers, not convolve 3 times!
    def __init__(self,a_fea_in,b_fea_in,nnn):
        '''
        a_fea_in: number of atom features
        b_fea_in: number of bond features
        nnn: number of nearest neighbors
        nconv: number of convolution times
        '''
        super(conv_layer,self).__init__()
        self.nnn = nnn
        self.a_fea_in = a_fea_in
        self.b_fea_in = b_fea_in
        self.b_update = torch.nn.Linear(a_fea_in*2+b_fea_in,b_fea_in)
        self.a_update = torch.nn.Linear(b_fea_in+a_fea_in,a_fea_in)
        self.bn1 = torch.nn.BatchNorm1d(b_fea_in)
        self.bn2 = torch.nn.BatchNorm1d(a_fea_in)

    def forward(self,nbr_idx,atom_fea,bond_fea):
        '''
        N is the number of atoms in the batch
        nbr_idx: indices of neighboring atoms with shape of N*nnn
        atom_fea: atom features with shape of N*a_fea_in
        bond_fea: bond features with shape of N*nnn*b_fea_in
        '''
        N = atom_fea.shape[0]
        bond_fea = torch.tanh(self.b_update(torch.cat((atom_fea.unsqueeze(1).expand(N,self.nnn,self.a_fea_in),atom_fea[nbr_idx],bond_fea),-1)))
        bond_fea = self.bn1(bond_fea.view(-1,self.b_fea_in)).view(N,self.nnn,self.b_fea_in)
        atom_fea = self.bn2(torch.tanh(self.a_update(torch.cat((torch.mean(bond_fea,axis=1),atom_fea),1))))# maybe sum pooling?
        torch.nn.Dropout()
        return atom_fea,bond_fea

class conv(torch.nn.Module):
    def __init__(self,n_atom_feature,n_bond_feature,nnn,a_embeded=20,b_embeded=30,nconv=3):
        '''
        n_atom_feature = number of original atom features
        n_bond_feature = number of original bond features
        a_embeded: number of atom features after embeding
        b_embeded: number of atom_update output neurons 
        nconv = number of convolutions
        nnn = number of nearest neighbors
        '''
        super(conv,self).__init__()
        self.nnn = nnn # number of nearest neighbors
        self.nconv = nconv
        self.convs = torch.nn.ModuleList([conv_layer(a_embeded,b_embeded,self.nnn) for _ in range(self.nconv)])
        self.n_bond_feature = n_bond_feature
        self.n_atom_feature = n_atom_feature
        self.bond_embed = torch.nn.Linear(n_bond_feature,b_embeded)
        self.atom_embed = torch.nn.Linear(n_atom_feature,a_embeded) 
        self.fc = torch.nn.Linear(a_embeded+b_embeded,1)

    def forward(self,gmap,atom,bonds,crystal_id):
        '''
       natom: number of atoms in the batch
       ncrystal: number of crystals in the batch, batch_size
       gmap:graph_map, the indicies of neighboring atoms
       atom: natom*n_atom_feature tensor
       bonds: natom*nnn*n_bond_feature tensor
       crystal_id: list, records the atom ids that belongs to each crystal.
       example: [torch.tensor([0,1,2]),torch.tensor([3,4,5,6,7])...] means that atom 0-2 belongs to 1st crystal;
       while atom 3-7 belongs to the second crystal. 
       The crystals with different number of atoms were connected along 1 direction to form a batch,
       thus the indices of neighboring atoms are shifted. 
        '''
        # embed
        natom = atom.shape[0] # number of atoms in the batch
        bonds = self.bond_embed(bonds)
        torch.nn.Dropout()
        atom = self.atom_embed(atom)
        torch.nn.Dropout()
        # convolution
        for convs in self.convs:
            atom,bonds = convs(gmap,atom,bonds)
        atom = torch.cat((torch.mean(bonds,axis=1),atom),1)
        # averaging the features of the atoms in each crystal to get the crystal representation:
        crystal_fea = torch.cat([torch.mean(atom[idx], dim=0, keepdim=True) for idx in crystal_id],dim=0)
        y = F.softplus(self.fc(crystal_fea))
        torch.nn.Dropout()
        return y
