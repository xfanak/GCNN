from model_crystal import *

data = torch.load('data.save')
label = torch.load('label.save')
label = [-i for i in label]
# split the dataset
# 26 batches in total, 22 of which serve as training data,
# the remaining are test data
train_data = data[:22]
train_label = label[:22]
test_data = data[22:]
test_label = label[22:]
model = conv(1,40,16,nconv=2)
criterion=torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters())
t0 = time.time()

for i in range(400):
    t = torch.randint(0,22,(1,))[0]
    crystal_id,atom_fea,nbr_fea,nbr_idx = train_data[t]
    y_predicted = model(nbr_idx,atom_fea,nbr_fea,crystal_id)
    loss = criterion(y_predicted,train_label[t].reshape(-1,1))
    print(i,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
t1 = time.time()
print(t1-t0)
record = []
for i in range(4):
    crystal_id,atom_fea,nbr_fea,nbr_idx = test_data[i]
    record.append(model(nbr_idx,atom_fea,nbr_fea,crystal_id))
yp = torch.cat(record,dim=0)
yt = torch.cat([i.view(-1,1) for i in test_label],dim=0)

plot(yp.detach(),yt,'o',label='Predicted')
plot(yt,yt,'o',label='True')
xlabel('Prediction')
ylabel('True formation energy')
legend()
show()
