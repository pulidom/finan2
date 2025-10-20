''' Pair trading  con todos los pares
       z-scores 
            Choice of beta calculacion: regresion / kalman filter
            Choice of averaging:
                  Using moving average window / exponential mean averaging / kalman filter  Todos los tiempos. Single set of parameters. Compara dos experimentos.    
'''
import numpy as np, os, copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from time import time

#from read_data import load_ts
from syn_data import load_sts
import arbitrage as ar
import plotting as gra
import cointegration as co
import utils

class cnf:
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'on'# 'kf' 'exp' 'on' 'off'
    Ntraining = 2*252 # length of the training period
    Njump = 84
    beta_win=121   #21
    zscore_win=41 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
#    nmax=None # number of companies to generate the pairs (-1 all, 10 for testing)
#    nsel=100# 100 # number of best pairs to select
    linver_betaweight=0
#    industry=['beverages'] #['oil'] # ['beverages']
    fname=f'tmp/syn_pair_{mtd}_' # fig filename
    shorten=0
    
# load data

nt= 5*252
ts = load_sts(nt=nt)


print('Nro de tiempos/dias',nt)
iini=0
res_l=[]
# select training period
for ilast in range(cnf.beta_win,nt,cnf.Njump):
#for ilast in range(cnf.Ntraining+cnf.Njump,cnf.Ntraining+2 * cnf.Njump,cnf.Njump):
    print(iini,ilast,ilast-iini)
    
    x,y=ts[:,iini:ilast]
    iini+=cnf.Njump

    t0 = time()
    res = ar.inversion(x,y,cnf,shorten=cnf.shorten)

    res_l.append(res)

    
res = {
    key: np.concatenate([res[key][cnf.beta_win:] for res in res_l],axis=0)
    for key in res_l[0].keys()  # Usa las keys del primer diccionario
    }

#res = utils.Results(**res) # mas elegante con objetos! :)
res = utils.dict2obj(**res) # mas elegante con objetos! :)

res.capital=np.zeros_like(res.retorno)
res.capital[0]=100
for i in range(res.retorno.shape[0]):
    res.capital[i,1:]= res.capital[i,0] * np.cumprod(1 + res.retorno[i,1:])


   
figfile=cnf.fname+'capital.png'
fig, ax = plt.subplots(1,1,figsize=(7,5))
ax.plot(res.capital[0],label='cap 5')
ax.plot(res.capital[1],label='cap 10')
ax.set(title='capital testing')
plt.tight_layout()
fig.savefig(figfile)
plt.close()



figfile=cnf.fname+'capital-largo-corto.png'
fig, ax = plt.subplots(3,1,figsize=(7,7))
ax[0].plot(res.largo.mean(-1).T)
ax[0].set(title='largo')
ax[1].plot(res.corto.mean(-1).T)
ax[1].set(title='corto')
ax[2].plot(res.capital.T)
ax[2].set(title='capital')
plt.tight_layout()
fig.savefig(figfile)
plt.close()


gra.plot_zscore(0,res,cnf.fname)
gra.plot_zscore(1,res,cnf.fname)
gra.plot_capital_single(0,res,cnf.fname)
gra.plot_capital_single(1,res,cnf.fname)

quit()


metrics = co.stats(res.assets,'log')
figfile=cnf.fname+f'scatters.png'
fig, ax = plt.subplots(2,3,figsize=(9,3))
ax[0,0].scatter(metrics.pvalue,metrics.johansen,s=10)
ax[0,0].set(xlabel='p-value',ylabel='johansen')
ax[0,1].scatter(metrics.hurst,metrics.half_life,s=10)
ax[0,1].set(xlabel='hurst',ylabel='half-life',ylim=[0,400])
ax[0,2].scatter(metrics.pvalue,res.capital[:,-1],s=10)
ax[0,2].set(xlabel='p-value',ylabel='ganancia')

ax[1,0].scatter(metrics.johansen,res.capital[:,-1],s=10)
ax[1,0].set(xlabel='johansen',ylabel='ganancia')
ax[1,1].scatter(metrics.hurst,res.capital[:,-1],s=10)
ax[1,1].set(xlabel='hurst',ylabel='ganancia')
ax[1,2].scatter(metrics.half_life,res.capital[:,-1],s=10)
ax[1,2].set(xlabel='half-life',ylabel='ganancia',xlim=[0,400])

plt.tight_layout()
fig.savefig(figfile)
plt.close()
