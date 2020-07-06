from model import *
from data import *
from pylab import plot,show,xlabel,ylabel,legend,hist2d,colorbar
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler

# functions
def mae(a,b):
    return torch.mean(torch.abs(a-b)).item()
def rho(t,p):
    a = np.sum((t - t.mean())*(p-p.mean()))
    b = np.sum((t - t.mean())**2) *np.sum( (p-p.mean())**2)
    return a/np.sqrt(b)

# dataset
Sitemode = True
tsfm = Crystal_embeding('atom_fea.save',root_dir='../train_1/p0_data/',fmat='.save')
crydata = CrystalDataset(id_prop_file='id_prop.save',root_dir='p_data/',transform=tsfm,preprocessed=True)
ntot = len(crydata) # total number of crystals
train_ratio = 0.96
train_size = int(ntot*train_ratio)
indices = list(range(ntot))
train_sampler = SubsetRandomSampler(indices[:train_size])
test_sampler=SubsetRandomSampler(indices[train_size:])
if Sitemode:
    collate_fn = collect_pool_site
else:
    collate_fn = collect_pool
train_loader = DataLoader(crydata,batch_size=3,collate_fn=collate_fn,sampler=train_sampler)
test_loader = DataLoader(crydata,batch_size=1,collate_fn=collect_pool_site,sampler=test_sampler)

# define model
model = GCNN(2,50,13,nconv=3,site=Sitemode)
criterion=torch.nn.L1Loss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
scheduler = MultiStepLR(optimizer, milestones=[100,200], gamma=0.8)
def train(nepochs):
    model.train()
    for epoch in range(nepochs):
        for data in train_loader:
            crystal_id,atom_fea,nbr_fea,nbr_idx= data[:-1]
            target = data[-1]
            y_predicted = model(nbr_idx,atom_fea,nbr_fea,crystal_id)
            loss = criterion(y_predicted,target)
            if epoch%10 ==0:
                print(epoch,loss.item(),mae(y_predicted.detach(),target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

def test(i):
    model.eval()
    cry_id = None
    with torch.no_grad():
        data = crydata[i]
        atom_fea,nbr_fea,nbr_idx= data['crystal_fea']
        target = data['property']
        y_predicted = model(nbr_idx,atom_fea,nbr_fea,cry_id)
    y_p = y_predicted.numpy().flatten()
    y_t = target.flatten()
    plot(y_t,y_t,'.',color='black',label='True')
    plot(y_t,y_p,'.',label='Predicted')
    legend()
    show()
    return np.abs(y_p-y_t).mean(),rho(y_t,y_p)
