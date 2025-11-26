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
from scipy import stats
#from functools import partial
import ot2
#import nns
from syn_data import load_sts

# Step 3: Setup and train
#nns.set_seed(40)

device='cpu'
n_train=400
n_val=100
batch_size=32
#normalize='gauss'#'minmax'#'gauss' # uniform in z for regression / gaussian for assets
#dat_type=['train','val','test']
#dat_spec = {'train' : ('train.npz', 0,n_train, batch_size, True),
#            'val' : ('val.npz', n_train,n_train + n_val, n_val, False),
#            'test' : ('test.npz', n_train + n_val, n_train + 2*n_val, n_val, False),}

class conf:
    seed = 42
#   loss = nns.mixed_loss2 #MSE_loss #mixed_loss  # Your NLL-based loss
    learning_rate = 5e-3
    n_epochs = 100
    patience = 15
    sexp = 'exp1'
    exp_dir = './tmp/'
    lvalidation = True
    warmup_epochs = 20
    layers = np.array([1, 32, 16,2])
#    activation = nn.GELU() #nn.ReLU() #nn.GELU() #nn.Tanh() 

nt= n_train + 2 * n_val
ts = load_sts(nt=nt,lopt=0,regime_length=nt,seed=43)
ts=ts[:2].T
print(ts.shape)

nts = (ts-ts.mean(0))/ts.std(0)

data = stats.norm.cdf(nts)
# Fit copulas
fitter = ot2.CopulaFitter(data)
results = fitter.fit_all_copulas()
    
# Print summary
print("\n" + "="*60)
print("RANKING OF COPULAS BY WASSERSTEIN DISTANCE")
print("="*60)
for i, result in enumerate(results, 1):
    param_str = f"θ={result['parameter']:.4f}" if result['parameter'] is not None else "N/A"
    print(f"{i}. {result['copula'].capitalize():15s} | {param_str:15s} | W₂={result['distance']:.6f}")
    
    # Plot comparison
fitter.plot_comparison(results)

#xydat=[]
#loaders=[]
#for dat_name in dat_type:
#
#    file, n0, n1, batch_size, shuffle = dat_spec[dat_name]
#    
#    dat = ts[n0:n1].T
#    dset = nns.DriveData(dat, device = device, normalize = normalize)
#
#    xydat.append([dset.x_data,dset.y_data]) # normalized
#    loader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
#    loaders.append(loader)
#
#
#NNmdl = nns.MeanVarianceNN(conf.layers, conf.activation).to(device)
##NNmdl = nns.BoundaryNet(NNmdl, boundary_width=0.1,min_sigma=0.1)
#NNmdl.apply(nns.init_weights)
#
## Entreno con datos reales
#best_net, loss_t, loss_v = nns.train(NNmdl, loaders[0], loaders[1], conf)
#
## Testing Dataset
#for i_batch, (input_test,target_test) in enumerate(loaders[2]):
#    pred_test = best_net(input_test)
#
## Smooth prediction
#train_x, train_y = xydat[0]
#train_x=train_x.squeeze().cpu().detach().numpy()
#train_y=train_y.squeeze().cpu().detach().numpy()
#x_smo=np.linspace(min(train_x), max(train_x),300)
#x_smo = torch.from_numpy(np.asarray(x_smo[:,None],dtype=np.float32)).to(device)
#pred_smo = best_net(x_smo)#,10)
#
def zscore(z,x):
    xpred = best_net(z)
    zsc= (x-xpred.T[0][:,None])/xpred.T[1][:,None]
    return zsc 
z_score= zscore(dset.x_data,dset.y_data)

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

plt.figure(figsize=(8,4))
z_score=z_score.cpu().detach().numpy()
plt.plot(z_score)
plt.xlabel(r'$t$');
plt.ylabel(r'$z-score$');
plt.savefig(f'{conf.exp_dir}/zcore.png')
