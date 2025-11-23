import numpy as np
#import matplotlib.pyplot as plt
import my_pyplot as plt
import torch, os, copy
from torch import nn
from torch.utils.data import Dataset
from torch.utils import data 
from time import time
#import optimal_transport as ot
import ot3
import nns
from numpy.polynomial import polynomial as P
from numpy.polynomial import polynomial as P
from functools import partial

# Step 3: Setup and train
nns.set_seed(40)

device='cpu'
n_train=500
n_val=500
batch_size=16
normalize='minmax' # uniform in z for regression / gaussian for assets
dat_type=['train','val','test']
dat_spec = {'train' : ('train.npz', n_train, batch_size, True),
            'val' : ('val.npz', n_val, n_val, False),
            'test' : ('test.npz', n_val, n_val, False),}

xydat=[]
loaders=[]
for dat_name in dat_type:

    file, nt, batch_size, shuffle = dat_spec[dat_name]
    
    #dat = Mdl.read_ts(conf.exp_dir + '/' + file, nt=nt)
    dat = nns.sin_sampler(nt)
    xydat.append(dat)

    dset = nns.DriveData(dat, device = device, normalize = normalize)

#    xydat.append([dset.x_data,dset.y_data])
    loader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    loaders.append(loader)

    
# Get polynomial boundaries from training data
train_x, train_y = xydat[0]
train_x=train_x.squeeze()
train_y=train_y.squeeze()

syn_x, syn_y, poly_coefs = nns.syndata_polynomial(train_x, train_y, 10000, degree=6, noise_scale=1)
dset = nns.DriveData([syn_x,syn_y], device = device, normalize = normalize)
loader_syn = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

# Network
layers = np.array([1, 64, 32,2])
activation = nn.GELU() #Tanh() #GELU()

NNmdl = nns.NNmodel(layers, activation).to(device)
#NNmdl.apply(nns.init_weights)

#base_net = nns.NNmodel(layers, activation).to(device)
#NNmdl = PolynomialBoundaryNet(base_net, y_left, y_right, boundary_width=0.02).to(device)

# Train with your existing setup
class conf:
    seed = 42
    loss = nns.mixed_loss #MSE_loss #mixed_loss  # Your NLL-based loss
    learning_rate = 5e-4
    n_epochs = 200
    patience = 80
    sexp = 'exp1'
    exp_dir = './tmp/'
    lvalidation = True
    warmup_epochs = 20


# Entreno con datos sinteticos
best_net, loss_t, loss_v = nns.train(NNmdl, loader_syn, loaders[1], conf)


# Entreno con datos reales
conf.lvalidation=True
conf.loss = nns.mixed_loss
best_net, loss_t, loss_v = nns.train(best_net, loaders[0], loaders[1], conf)

# Testing
for i_batch, (input_dat,target_dat) in enumerate(loaders[2]):
    pred = best_net(input_dat)#,10)

# Smooth prediction
xs=np.linspace(0,1,500)
x_dat = torch.from_numpy(np.asarray(xs[:,None],dtype=np.float32)).to(device)
pred = best_net(x_dat)#,10)

pred=pred.cpu().detach().numpy()
x_dat=x_dat.cpu().detach().numpy()

input_dat=input_dat.cpu().detach().numpy()
target_dat=target_dat.cpu().detach().numpy()

def ot(input_dat,target_dat):
    #y_sample,  s, U, Vt, Qz, Bz, By, centers_z, lambda_val = ot.ot_barycenter_solver(target_dat.squeeze(), input_dat.squeeze(), n_iter=500)

    y_sample, state = ot3.ot_barycenter_solver(target_dat.squeeze(), input_dat.squeeze(), 
                                               method='gaussian',  
                                               n_centers_x=20,     
                                               n_centers_z=15,
                                               bandwidth_x=1.5,   
                                               bandwidth_z=1,
                                               aggressive=False,
                                               n_iter=3000,
                                               verbose=True)

    #y_sample, state = ot3.ot_barycenter_solver(target_dat.squeeze(), input_dat.squeeze(), 
    #                                method='polynomial',
    #                                aggressive=True,
    #                                n_iter=2000,
    #                                verbose=True)
    xot=np.linspace(0,1,100)
    x_sample=[]
    for z in xot:
        x_sample.append (state.simulate_conditional(y_sample, z_target=z))
    #    x_sample.append( ot.simulate_conditional(y_sample, x, Qz, Bz, By, centers_z, lambda_val, s, U, Vt) )

    x_sample=np.array(x_sample)
    xmean=x_sample.mean(1)
    xstd= x_sample.std(1)
    
    return x_sample,xmean,xstd


    
# plot syntethic data
x_smooth = np.linspace(0, 1, 200)
y_poly = P.polyval(x_smooth, poly_coefs)
plt.figure(figsize=(8,4))
plt.plot(syn_x,syn_y, '.')
plt.plot(x_smooth,y_poly)
plt.plot(train_x,train_y, '.')
plt.xlabel(r'$z$');
plt.ylabel(r'$x$');
plt.savefig(f'{conf.exp_dir}/polyreg.png')

plt.figure(figsize=(6,4))
plt.plot(range(len(loss_t)),loss_t,color='C0')
plt.plot(range(len(loss_v)),loss_v,color='C1')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.savefig(f'{conf.exp_dir}/loss.png')

plt.figure(figsize=(8,4))
plt.plot(input_dat,target_dat, '.')
plt.plot(x_dat,pred.T[0], '.')
plt.fill_between(x_dat.T[0],pred.T[0] -  2* pred.T[1], pred.T[0] + 2* pred.T[1], color='gray', alpha=0.2)
plt.xlabel(r'$z$');
plt.ylabel(r'$x$');
plt.savefig('{conf.exp_dir}/uncertaintyFnX.png')

plt.figure(figsize=(8,4))
plt.plot(x_dat,pred.T[1], '.')
plt.xlabel(r'$z$');
plt.ylabel(r'$x$');
plt.savefig('{conf.exp_dir}/sigma.png')

if False:
    x_sample,xmean,xstd = ot(input_dat,target_dat)
    plt.figure(figsize=(8,4))
    plt.plot(input_dat,target_dat, '.')
    #plt.plot(x_dat,pred.T[0], '.')
    print(input_dat.shape)
    plt.fill_between(xot,xmean -  2* var, xmean + 2* var, color='gray', alpha=0.2)
    plt.xlabel(r'$z$');
    plt.ylabel(r'$x$');
    plt.savefig('ot.png')
