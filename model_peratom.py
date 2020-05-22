# This model handles per-atom quantities only, like propensity, magnetic moment, potential energy etc.

import torch
import torch.nn.functional as F

class conv(torch.nn.Module):
    def __init__(self,batch_size,n_atom_feature,n_bond_feature,nnn,h1=30,h2=30,nconv=3):
        '''
        h1: number of bond_update output neurons
        h2: number of atom_update output neurons 
        nconv = number of convolutions
        nnn = number of nearest neighbors
        '''
        super(conv,self).__init__()
        self.nnn = nnn # number of nearest neighbors
        self.batch_size = batch_size
        self.nconv = nconv
        self.n_atom_feature = n_atom_feature
        self.h2 = h2
        self.bond_embed = torch.nn.Linear(n_bond_feature+2*n_atom_feature,h1) # cat the bond features and corresponding atom features into 1 vector
        self.atom_embed = torch.nn.Linear(h1 + n_atom_feature,h2) # h1+n_atom_feature = number of atom features after embeding
        self.bond_update = torch.nn.Linear(h1+2*h2,h1)
        self.atom_update = torch.nn.Linear(h1+h2,h2)
        self.fc = torch.nn.Linear(h2,1)

    def forward(self,gmap,atom,bonds):
        '''
       gmap:graph_map, the indicies of neighboring atoms
       atom: batch_size*n_atom_feature tensor
       bonds: batch_size*nnn*n_bond_feature tensor
        '''
        # embed
        b0 = torch.cat((atom.unsqueeze(1).expand(self.batch_size,self.nnn,self.n_atom_feature),atom[gmap],bonds),-1)
        bonds = torch.tanh(self.bond_embed(b0))
        torch.nn.Dropout()
        atom = F.relu(self.atom_embed(torch.cat((torch.mean(bonds,axis=1),atom),1))) # average pooling on bond features`
        torch.nn.Dropout()
        # convolution
        for k in range(self.nconv):
            bonds = torch.tanh(self.bond_update(torch.cat((atom.unsqueeze(1).expand(self.batch_size,self.nnn,self.h2),atom[gmap],bonds),-1)))
            torch.nn.Dropout()
            atom = F.relu(self.atom_update(torch.cat((torch.mean(bonds,axis=1),atom),1)))
            torch.nn.Dropout()
        y = F.softplus(self.fc(atom))
        return y

