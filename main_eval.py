#!/usr/bin/python3
"""
Lee la red neuronal entrenada para L63
"""
import sys; sys.path.insert(0, '../')
import numpy as np
import my_pyplot as plt
import torch
from syn_data import load_sts
from nn.datagen import create_dataloaders
import nn.datagen as dat
import nn.net as net
import nn.lossfn as lossfn
from nn.train import train
import utils
import conf_rnn as cnf

## conf para el modelo dinamico
#class cnf_cls:
#    pathdat='dat/'
##    tipo='asset' # 'asset', 'return', 'log_return', 'log'
##    mtd = 'on'# 'kf' 'exp' 'on' 'off'
#    tipo='asset'#log' #'asset' # 'asset', 'return', 'log_return', 'log'
#    mtd = 'nn'# 'kf' 'exp' 'on' 'off'
#    n_train = 10_000
#    n_val = 5_000
#    Njump = 10 # peque~no para el entrenamiento
#    beta_win=121 #*121   #21
#    zscore_win=41 #11
#    batch_size=256
#    nt_start=beta_win
#    nt_window=beta_win+zscore_win
#    sigma_co=1.5 # thresold to buy
#    sigma_ve=0.1 # thresold to sell
##    nmax=None # number of companies to generate the pairs (-1 all, 10 for testing)
##    nsel=100# 100 # number of best pairs to select
#    linver_betaweight=0
##    industry=['beverages'] #['oil'] # ['beverages']
#    fname=f'tmp/syn_pair_{mtd}_' # fig filename
#    shorten=0
#    
#class cnf_train: 
#    loss = lossfn.loss_fn #nn.MSELoss() #nn.logGaussLoss #SupervLoss # GaussLoss
#    batch_size=cnf_cls.batch_size
#    n_epochs = 200
#    learning_rate = 1e-3
#    exp_dir = 'dat/rnn/'
#    lvalidation = True 
#    patience = 30
#    sexp = 'syn_rnn'
#
## Define parametros de la red
#class cnf_net:
#    hidden_dim=40
#    layers=3
#    input_dim=2
#    output_dim=1
#    dropout=0.1
#    device='cpu'
#    nt_start=100 # warm-up times/ input_times
#    nt_window=cnf_cls.nt_window
#    n_samples = 100
#
#cnf = cnf.cnf_cls()

nt=cnf.dat.n_train+2*cnf.dat.n_val
ts = load_sts(nt=nt,lopt=0,regime_length=nt,seed=53)#252)
x0,y0,nret0_x,nret0_y = utils.select_variables(ts[0],ts[1],tipo=cnf.dat.tipo)
ts = np.vstack([x0, y0])
[loader_train, loader_val, loader_test] = create_dataloaders(ts,cnf.dat)

net_file = cnf.train.exp_dir + f'/model_{cnf.train.sexp}_best.ckpt'
Net =torch.load(net_file,map_location=torch.device(cnf.net.device))

for i_batch, (input_dat,target_dat) in enumerate(loader_test):
    input_dat=input_dat.transpose(0,1)
    mu_pred,sigma_pred = Net.train_step(input_dat)

    
mu = mu_pred.transpose(0,1).cpu().detach().numpy().squeeze()
sigma = sigma_pred.transpose(0,1).cpu().detach().numpy().squeeze()
target_dat = target_dat.cpu().detach().numpy().squeeze()
input_dat = input_dat.transpose(0,1).cpu().detach().numpy().squeeze()
figfile=cnf.dat.fname+'prediction.png'
t=np.arange(mu.shape[1])
print(t.shape)
print(mu.shape)
print(sigma.shape)
print(target_dat.shape)

fig, ax = plt.subplots(3,1,figsize=(7,7))
ax[0].fill_between(t,mu[50] - sigma[50], mu[50] + sigma[50], color='gray', alpha=0.2)
ax[0].plot(t,mu[50])
ax[0].plot(t,target_dat[50])
ax[0].plot(t,input_dat[50])
ax[1].fill_between(t,mu[100] - sigma[100], mu[100] + sigma[100], color='gray', alpha=0.2)
ax[1].plot(t,mu[100])
ax[1].plot(t,target_dat[100])
ax[1].plot(t,input_dat[100])
ax[2].fill_between(t,mu[200] - sigma[200], mu[200] + sigma[200], color='gray', alpha=0.2)
ax[2].plot(t,mu[200])
ax[2].plot(t,target_dat[200])
ax[2].plot(t,input_dat[200])
plt.tight_layout()
fig.savefig(figfile)
plt.close()

