#!/usr/bin/python3
"""
Entrenamiento de una red neuronal tipo DeepAR para finanzas con generacion automatica
"""
#import sys; sys.path.insert(0, '../finan2')
import numpy as np
import my_pyplot as plt
import torch
import nn.datagen as dat
from syn_data import load_sts
import nn.net as net
from nn.train import train
# conf para el modelo dinamico
import conf_rnn as conf

# Define parametros de los datos
nt=conf.dat.n_train+2*conf.dat.n_val   
ts = load_sts(nt=nt,lopt=0,regime_length=252,seed=53)#252)

print(ts.shape)
# Generate training dataset
[loader_train, loader_val, loader_test] = dat.create_dataloaders(ts,conf.dat)
# Neural net
Net = net.Net(conf.net)
# Training
best_net,loss_t,loss_v = train( Net,loader_train,loader_val,conf.train )

plt.figure(figsize=(6,4))
plt.plot(range(len(loss_t)),loss_t,color='C0')
plt.plot(range(len(loss_v)),loss_v,color='C1')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.savefig(conf.exp_dir+f'/loss.png')

#quit()
#[loader_test]  = create_dataloaders(conf,dat_type=['test'])
#
#for i_batch, (input_dat,target_dat) in enumerate(loader_test):
#    mu_pred,sigma_pred = best_net.train_step(input_dat)
#
#mu_pred = mu_pred.cpu().detach().numpy()
#sigma_pred = sigma_pred.cpu().detach().numpy()
#target_dat = target_dat.cpu().detach().numpy()
#figfile=conf.exp_dir+'prediction.png'
#t=np.arange(mu_pred.shape[1])
#fig, ax = plt.subplots(3,1,figsize=(7,7))
#ax[0].plot(t,mu_pred[50])
#ax[0].plot(t,target_dat[50])
#ax[1].plot(t,mu_pred[100])
#ax[1].plot(t,target_dat[100])
#ax[2].plot(t,mu_pred[200])
#ax[2].plot(t,target_dat[200])
#plt.tight_layout()
#fig.savefig(figfile)
#plt.close()
#
