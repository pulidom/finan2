''' Pair trading  con todos los pares
       z-scores 
            Choice of beta calculacion: regresion / kalman filter
            Choice of averaging:
                  Using moving average window / exponential mean averaging / kalman filter  Todos los tiempos. Single set of parameters. Compara dos experimentos.    
'''
import numpy as np, os, copy
#import matplotlib.pyplot as plt
import my_pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from time import time

#from read_data import load_ts
from syn_data import load_sts
from nn.datagen import create_dataloaders
import nn.net as net
import nn.lossfn as lossfn
import nn.train as train
import arbitrage as ar
import plotting as gra
import cointegration as co
import utils
import plotting
#from plotting import vertical_bar

class cnf_cls:
    pathdat='dat/'
#    tipo='asset' # 'asset', 'return', 'log_return', 'log'
#    mtd = 'on'# 'kf' 'exp' 'on' 'off'
    tipo='log' #asset'#log' #'asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'nn'# 'kf' 'exp' 'on' 'off'
    n_train = 10_000
    n_val = 5_000
    Njump = 10 # peque~no para el entrenamiento
    beta_win=121 #*121   #21
    zscore_win=41 #11
    batch_size=256
    nt_start=beta_win
    nt_window=beta_win+zscore_win
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
#    nmax=None # number of companies to generate the pairs (-1 all, 10 for testing)
#    nsel=100# 100 # number of best pairs to select
    linver_betaweight=0
#    industry=['beverages'] #['oil'] # ['beverages']
    fname=f'tmp/syn_pair_{mtd}_' # fig filename
    shorten=0
    
class cnf_train: 
    loss = lossfn.loss_fn #nn.MSELoss() #nn.logGaussLoss #SupervLoss # GaussLoss
    batch_size=cnf_cls.batch_size
    n_epochs = 200
    learning_rate = 1e-3
    exp_dir = 'dat/rnn'
    lvalidation = True 
    patience = 30
    sexp = 'syn_rnn'

# Define parametros de la red
class cnf_net:
    hidden_dim=40
    layers=3
    input_dim=2
    output_dim=1
    dropout=0.1
    device='cpu'
    nt_start=100 # warm-up times/ input_times
    nt_window=cnf_cls.nt_window
    n_samples = 100

    
cnf_co = cnf_cls(); cnf = cnf_cls()
cnf_co= copy.deepcopy(cnf)
cnf_co.tipo='asset'
cnf_co.mtd='on'

# load data
nt=cnf.n_train+2*cnf.n_val
#nt= 10*252

ts = load_sts(nt=nt,lopt=0,regime_length=nt,seed=43)#252)
#figfile=cnf.fname+'prediction.png'
#t=np.arange(ts[0].shape[0])
#fig, ax = plt.subplots(3,1,figsize=(7,7))
#ax[0].plot(t[0:1_000],ts[0,0:1_000])
#ax[0].plot(t[0:1_000],ts[1,0:1_000])
#ax[1].plot(t[10_000:12_000],ts[0,10_000:12_000])
#ax[1].plot(t[10_000:12_000],ts[1,10_000:12_000])
#ax[2].plot(t[18_000:20_000],ts[0,18_000:20_000])
#ax[2].plot(t[18_000:20_000],ts[1,18_000:20_000])
#plt.tight_layout()
#fig.savefig(figfile)
#plt.close()
#ts = load_sts(nt=nt,lopt=0,regime_length=nt,seed=43)#252)
#figfile=cnf.fname+'prediction.png'
#t=np.arange(ts[0].shape[0])
#fig, ax = plt.subplots(3,1,figsize=(7,7))
#ax[0].plot(t[0:1_000],ts[0,0:1_000])
#ax[0].plot(t[0:1_000],ts[1,0:1_000])
#ax[1].plot(t[10_000:12_000],ts[0,10_000:12_000])
#ax[1].plot(t[10_000:12_000],ts[1,10_000:12_000])
#ax[2].plot(t[18_000:20_000],ts[0,18_000:20_000])
#ax[2].plot(t[18_000:20_000],ts[1,18_000:20_000])
#plt.tight_layout()
#fig.savefig(figfile)
#plt.close()

x0,y0,nret0_x,nret0_y = utils.select_variables(ts[0],ts[1],tipo=cnf.tipo)


ts = np.vstack([x0, y0])
[loader_train, loader_val, loader_test] = create_dataloaders(ts,cnf)


# Neural net
Net = net.Net(cnf_net)
# Training
best_net,loss_t,loss_v = train.train( Net,loader_train,loader_val,cnf_train )

test_batch= next(iter(loader_test))
in0,out0=test_batch
print(in0.shape)
figfile=cnf.fname+'prediction-test.png'
fig, ax = plt.subplots(3,1,figsize=(7,7))
fig, ax = plt.subplots(3,1,figsize=(7,7))
ax[0].plot(in0[50])
ax[0].plot(out0[50])
#ax[1].plot(t,mu_pred[100])
#ax[1].plot(t,target_dat[100])
#ax[2].plot(t,mu_pred[200])
#ax[2].plot(t,target_dat[200])
plt.tight_layout()
fig.savefig(figfile)

for i_batch, (input_dat,target_dat) in enumerate(loader_test):

    mu_pred,sigma_pred = best_net.train_step(input_dat)

#print(mu.shape)
#quit()
### reseteo del Njump
#cnf.Njump=84 

# hago todas las predicciones
#mu_pred,sigma_pred,target = train.test(loader_test,best_net,deterministic=True)

mu_pred = mu_pred.cpu().detach().numpy()
sigma_pred = sigma_pred.cpu().detach().numpy()
target_dat = target_dat.cpu().detach().numpy()
print(mu_pred.shape)
print(sigma_pred.shape)
figfile=cnf.fname+'prediction.png'
t=np.arange(mu_pred.shape[1])
fig, ax = plt.subplots(3,1,figsize=(7,7))
#ax[0].plot(t,mu_pred[50])
ax[0].plot(t,target_dat[50])
#ax[1].plot(t,mu_pred[100])
ax[1].plot(t,target_dat[100])
#ax[2].plot(t,mu_pred[200])
ax[2].plot(t,target_dat[200])
plt.tight_layout()
fig.savefig(figfile)
plt.close()

print('llego')
quit()
## select testing period
#nt0=cnf.n_train+cnf.n_val
#x0=x0[nt0:];y0=y0[nt0:]
#nret_x0=x0[nt0:];nret_y0=y0[nt0:]
#
#iini=0
#res_l=[]; res_co_l=[]
## select training period
#for ilast in range(cnf.beta_win+cnf.Njump,nt,cnf.Njump):### tengo que trabajar directamente con la red. Me podria predecir todo y luego voy haciendo las inversiones para cada tiempo???
#    print(iini,ilast,ilast-iini)
#    
#    x,y=x0[iini:ilast],y0[iini:ilast]
#    nret_x,nret_y=nret_x0[iini:ilast],nret_y0[iini:ilast]
#    
#    iini+=cnf.Njump
#
#    t0 = time()
#    #
#    mu_pred,sigma_pre, sample = 
#    # prediccion
#    zscore=XXXXXX
#    ### compute z-score
#    inversion_zscore(zscore0,nret_x,nret_y,cnf)
#
#    ### invierte
#    
#    res = ar.inversion(x,y,cnf,shorten=cnf.shorten)
#    res['asset_x']=x
#    res['asset_y']=y
#    res_l.append(res)
#
#    res_co = ar.inversion(x,y,cnf_co,shorten=cnf.shorten)
#    res_co_l.append(res_co)
#
#
#       
##ret1 = np.concatenate([res['retorno'][cnf.beta_win:] for res in res_l],axis=0)
#res = {
#    key: np.concatenate([res[key][cnf.beta_win:] for res in res_l],axis=0)
#    for key in res_l[0].keys()  # Usa las keys del primer diccionario
#    }
#res_co = {
#    key: np.concatenate([res[key][cnf.beta_win:] for res in res_co_l],axis=0)
#    for key in res_co_l[0].keys()  # Usa las keys del primer diccionario
#    }
#
##for key in res_l[0].keys():  # Usa las keys del primer diccionario
##    print(key)
#
#res = utils.dict2obj(**res) # mas elegante con objetos! :)
#res_co = utils.dict2obj(**res_co) # mas elegante con objetos! :)
#
#res.capital=np.zeros_like(res.retorno)
#res.capital[0]=100
#res.capital= res.capital[0] * np.cumprod(1 + res.retorno)
#
#res_co.capital=np.zeros_like(res_co.retorno)
#res_co.capital[0]=100
#res_co.capital= res_co.capital[0] * np.cumprod(1 + res_co.retorno)
#
##res.asset_x=x
##res.asset_y=y
#
#t=np.arange(res.asset_x.shape[0])/252
#   
#figfile=cnf.fname+'scatter.png'
#fig, ax = plt.subplots(1,1,figsize=(7,5))
#scatter = plt.scatter(res.asset_x, res.asset_y, c=t, cmap='viridis', s=10, alpha=0.7)
#plt.colorbar(scatter, label='Time')
#
##ax.plot(res.asset_x,res.asset_y,'.',label='cap 5')
##ax.plot(res.capital[0],label='cap 5')
##ax.plot(res.capital[1],label='cap 10')
#ax.set(title='Scatterplot activos')
#plt.tight_layout()
#fig.savefig(figfile)
#plt.close()
#   
#t=np.arange(res.capital.shape[0])/252
#   
#figfile=cnf.fname+'capital.png'
#fig, ax = plt.subplots(1,1,figsize=(7,5))
#ax.plot(t,res.capital,label='Tabak')
#ax.plot(t,res_co.capital,label='Cointegration')
##ax.plot(res.capital[0],label='cap 5')
##ax.plot(res.capital[1],label='cap 10')
#ax.set(title='capital testing')
#ax.legend()
#plt.tight_layout()
#fig.savefig(figfile)
#plt.close()
#
#
#t=np.arange(res.zscore.shape[0])/252
#figfile=cnf.fname+'zscores.png'
#fig, ax = plt.subplots(2,1,figsize=(7,7))
#ax[0].plot(t,res.zscore)
#plotting.vertical_bar([ax[0]],res.compras,res.ccompras)
#ax[1].plot(t,res_co.zscore)
#plotting.vertical_bar([ax[1]],res_co.compras,res_co.ccompras)
#plt.tight_layout()
#fig.savefig(figfile)
#plt.close()


