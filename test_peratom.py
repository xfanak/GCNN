import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from pylab import plot,show,xlabel,ylabel,legend
from model_peratom import conv

# load the training data and scale the features
gmap = torch.from_numpy(np.load('graph_map.npy'))
bonds = np.load('exp_dis.npy')
scaler = StandardScaler().fit(bonds.reshape(-1,50)) # scale makes the model easir to train
bonds = torch.from_numpy(scaler.transform(bonds.reshape(-1,50)).reshape(5,4000,20,50)).float()
xall = torch.from_numpy(np.load('atom_feature.npy')).float()
yall = -torch.from_numpy(np.loadtxt('pe.dat')).float().reshape(5,4000)
train_atom = xall[0:4]
train_y = yall[0:4,:]
test_atom = xall[-1]
test_y = yall[-1]
bonds_train = bonds[0:4,:,:,:]
bonds_test = bonds[-1]

model = conv(4000,3,50,20,nconv=1) # initialize the model
criterion=torch.nn.MSELoss(reduction='sum') # define loss function
optimizer = torch.optim.Adam(model.parameters()) # define optimizer

# We train 4,000 epochs
for i in range(4000):
    t = torch.randint(0,4,(1,))[0]
    y_predicted = model(gmap[t],train_atom[t],bonds_train[t])
    loss = criterion(y_predicted,train_y[t].reshape(4000,1))
    print(i,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plot(yp.detach().numpy(),test_y.numpy(),label='Predicted')
plot(test_y.numpy(),test_y.numpy(),'o',label='True')
xlabel('Prediction')
ylabel('True potential energy')
legend()
show()
errors = yp.detach().flatten()-test_y.flatten()
print('The sumation of MSE error is:',sum(yp**2))
