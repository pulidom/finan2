import numpy as np
#import matplotlib.pyplot as plt
import my_pyplot as plt
import torch, os, copy
from torch import nn
from torch.utils.data import Dataset
from torch.utils import data
from time import time
#import optimal_transport as ot
from numpy.polynomial import polynomial as P
from functools import partial
#import ot3
import nns
from syn_data import load_sts

# Step 3: Setup and train
nns.set_seed(40)

device='cpu'
n_train=1000
n_val=1000
batch_size=8
normalize='gauss' # uniform in z for regression / gaussian for assets
dat_type=['train','val','test']
dat_spec = {'train' : ('train.npz', 0,n_train, batch_size, True),
            'val' : ('val.npz', n_train,n_train + n_val, n_val, False),
            'test' : ('test.npz', n_train + n_val, n_train + 2*n_val, n_val, False),}

class conf:
    seed = 42
    loss = nns.mixed_loss2 #MSE_loss#mixed_loss #MSE_loss #mixed_loss  # Your NLL-based loss
    learning_rate = 5e-3
    n_epochs = 30
    patience = 15
    sexp = 'exp1'
    exp_dir = './tmp/'
    lvalidation = True
    warmup_epochs = 20
    layers = np.array([1, 32, 16,2])
    activation = nn.GELU() #Tanh() #GELU()

nt= n_train + 2 * n_val
ts = load_sts(nt=nt,lopt=0,regime_length=nt,seed=43)
ts=ts[:2].T

xydat=[]
loaders=[]
for dat_name in dat_type:

    file, n0, n1, batch_size, shuffle = dat_spec[dat_name]
    
    dat = ts[n0:n1].T
    dset = nns.DriveData(dat, device = device, normalize = normalize)

    xydat.append([dset.x_data,dset.y_data]) # normalized
    loader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    loaders.append(loader)



# Get polynomial boundaries from training data
train_x, train_y = xydat[0]
train_x=train_x.squeeze().cpu().detach().numpy()
train_y=train_y.squeeze().cpu().detach().numpy()

syn_x, syn_y, poly_coefs = nns.syndata_polynomial(train_x, train_y, 10000, degree=1, noise_scale=1)

# plot syntethic data
x_smooth = np.linspace(min(train_x), max(train_x), 200)
y_poly = P.polyval(x_smooth, poly_coefs)
plt.figure(figsize=(8,4))
#plt.plot(syn_x,syn_y, '.')
plt.plot(train_x,train_y, '.')
plt.plot(x_smooth,y_poly)
plt.xlabel(r'$z$');
plt.ylabel(r'$x$');
plt.savefig(f'{conf.exp_dir}/polyreg.png')




syn_x, syn_y, poly_coefs = nns.syndata_polynomial(train_x, train_y, 10000, degree=6, noise_scale=1)
dset = nns.DriveData([syn_x,syn_y], device = device, normalize = normalize)
loader_syn = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)


NNmdl = nns.NNmodel(conf.layers, conf.activation).to(device)
#NNmdl.apply(nns.init_weights)

# Entreno con datos sinteticos
best_net, loss_t, loss_v = nns.train(NNmdl, loader_syn, loaders[1], conf)


# Entreno con datos reales
conf.lvalidation=True
conf.n_epochs=100
#conf.loss = nns.mixed_loss
conf.learning_rate = 1e-3
best_net, loss_t, loss_v = nns.train(best_net, loaders[0], loaders[1], conf)

# Testing Dataset
for i_batch, (input_test,target_test) in enumerate(loaders[2]):
    pred_test = best_net(input_test)#,10)

# Smooth prediction
x_smo=np.linspace(min(train_x), max(train_x),300)
x_smo = torch.from_numpy(np.asarray(x_smo[:,None],dtype=np.float32)).to(device)
pred_smo = best_net(x_smo)#,10)


def loss_plot(loss_t,loss_v):
    plt.figure(figsize=(6,4))
    plt.plot(range(len(loss_t)),loss_t,label='Training')#,color='C0')
    plt.plot(range(len(loss_v)),loss_v,label='Testing')#,color='C1')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.legend()
    plt.savefig(f'{conf.exp_dir}/loss.png')
    
def pred_plot(input_dat,target_dat,pred,x_smo,pred_smo):
    input_dat=input_dat.cpu().detach().numpy()
    target_dat=target_dat.cpu().detach().numpy()
    pred=pred.cpu().detach().numpy()
    pred_smo=pred_smo.cpu().detach().numpy()
    x_smo=x_smo.cpu().detach().numpy()

    plt.figure(figsize=(8,4))
    plt.plot(input_dat,target_dat, '.',alpha=0.3,label='Target')
#    plt.plot(input_dat,pred, '.',alpha=0.3,label='Prediction')
    plt.plot(x_smo,pred_smo.T[0])#, '.')
    plt.fill_between(x_smo.T[0],pred_smo.T[0] -  2* pred_smo.T[1], pred_smo.T[0] + 2* pred_smo.T[1], color='gray', alpha=0.2)
    plt.xlabel(r'$z$');
    plt.ylabel(r'$x$');
    plt.savefig(f'{conf.exp_dir}/prediction.png')

    
loss_plot(loss_t,loss_v)
pred_plot(input_test,target_test,pred_test,x_smo,pred_smo)
