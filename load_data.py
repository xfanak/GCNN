import numpy as np
from pylab import plot,show

class load_pos(object):
    '''
    for VASP format: Direct coordinates only
    '''
    def load_poscar(self,fname):
        f = open(fname,'r')
        f.readline()
        scale = float(f.readline())
        lattice = np.array([f.readline().split() for i in range(3)])
        self.lattice = scale*lattice.astype(float)
        f.readline()
        self.N_pertype = [int(j) for j in f.readline().split()]
        self.ntypes = len(self.N_pertype)
        self.Natom = sum(self.N_pertype)
        f.readline()
        self.lc = np.linalg.norm(self.lattice,axis=1)
        self.pos = np.array([[float(i) for i in f.readline().split()] for j in range(self.Natom)])*self.lc
        f.close()

    def load_dump(self,fname): # load information from a LAMMPS dump file
        with open(fname,'r') as f:
            head = [next(f) for x in range(4)]
            ntot = int(head[3])
            pos = []
            lattice = []
            for line in f:
                li = line.split()
                if len(li) == 2 and not line.startswith('I'):
                    for x in li:
                        lattice.append(float(x))
                elif len(li) == 5:
                    for x in li:
                        pos.append(float(x))
        lattice = np.array(lattice).reshape(-1,2) # 3*nframe*2 array
        lattice = lattice[::3]
        self.lc = lattice[:,1] - lattice[:,0]
        self.ntot = ntot
        self.nframe = self.lc.shape[0]
        pos = np.array(pos).reshape(-1,5)[:,2:]
        self.pos = pos.reshape(self.nframe,self.ntot,3)

def get_neighbor(pos,l,ncut=21):
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

def neighbor_pairs(gmap):
    n,l = gmap.shape
    l = l - 1
    index0 = []
    index1 = []
    for i in range(n):
        index0.append(np.repeat(gmap[i,0],l))
        index1.append(gmap[i,1:])
        #record.append(np.array(list(zip(np.repeat(gmap[i,0],l),gmap[i,1:]))))
    return np.array(index0),np.array(index1)

def expand_dis(dis,l=4.5,nbin=50,sigma=0.5):
    m,n = dis.shape
    r0 = np.linspace(0,l,nbin,False)
    dis = dis.reshape(-1,1)
    expo = (dis - r0)**2/sigma**2
    return np.exp(-expo).reshape(m,n,nbin)

l = load_pos()
l.load_dump('dump.dat')
gmap = []
expdis = []
for i in range(5):
    graph_map,dis = get_neighbor(l.pos[i],l.lc[i])
    dis = expand_dis(dis)
    gmap.append(graph_map)
    expdis.append(dis)
gmap = np.array(gmap)
expdis = np.array(expdis)
#i0,i1 = neighbor_pairs(graph_map)
#np.save('graph_map.npy',gmap)
#np.save('exp_dis',expdis)

#xall = np.zeros((5,4000,3)) # 3 features for each type: type, radius, mass
#xall[:,:3200,0] = 0
#xall[:,3200:,0] = 1
#xall[:,:3200,1] = 1
#xall[:,3200:,1] = 0.88
#xall[:,:3200,2] = 1
#xall[:,3200:,2] = 1
#np.save('atom_feature.npy',xall)
