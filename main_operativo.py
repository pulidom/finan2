''' Pair trading  con todos los pares
       z-scores 
            Choice of beta calculacion: regresion / kalman filter
            Choice of averaging:
                  Using moving average window / exponential mean averaging / kalman filter  Todos los tiempos. Single set of parameters. Compara dos experimentos.    
'''
### Para seleccionar los pares en este codigo se puede usar capital,pval,sharpe,dias positivos
import numpy as np, os, copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from time import time

from read_data import load_ts
import arbitrage as ar
import plotting as gra
import cointegration as co
import utils
# Njump     = ({21,42,63},84,126,252)
# Ntraining = ({126},252,504,756,1008)
class cnf:
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'on'# 'kf' 'exp' 'on' 'off'
    Ntraining = 131 # length of the training period
    Njump = 80
    beta_win=121   #21
    zscore_win=41 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
    nmax=-1 # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=100# 100 # number of best pairs to select
    fname=f'tmp/all_pair_{mtd}_/pruebas_opt_pval' # fig filename
    linver_betaweight=0
    #industry='oil'
    industry='marine' 
    shorten=0
    
# load data
day,date,price,company = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)


caps = [[] for _ in range(6)]  

nt=price.shape[1]
iini=0
# select training period
for ilast in range(cnf.Ntraining+cnf.Njump,nt,cnf.Njump):
    print(iini,ilast,ilast-iini)
    
    assets_tr=price[:cnf.nmax,iini:ilast]

    print(assets_tr.shape)
    iini+=cnf.Njump
    
 
    t0 = time()
    res = ar.all_pairs(assets_tr,company[:cnf.nmax],cnf)
    print('Tiempo:  ',time()-t0)

    res2=copy.deepcopy(res)
    res3=copy.deepcopy(res)
    res4=copy.deepcopy(res)
    # Select nsel best pairs
    idx = np.argsort(res.capital[:,ilast-cnf.Njump-iini])[::-1][:cnf.nsel]
    metrics = co.all_pairs_stats(assets_tr[:,:ilast-cnf.Njump],company,'asset')

    idx = np.argsort(metrics.half_life)[:cnf.nsel]

    res.reorder(idx) # ordeno todo los resultados segun el capital
    #cap_pred=utils.returns_from(res.capital,cnf.Ntraining)
    metrics = co.all_pairs_stats(assets_tr[:,:ilast-cnf.Njump],company,'asset')
    idx = np.argsort(metrics.pvalue)[:cnf.nsel]
    res2.reorder(idx) # ordeno todo los resultados segun el p-value
    #cap_pred2=utils.returns_from(res2.capital,cnf.Ntraining)
    sharpe = np.mean(res.retorno, axis=1) / np.std(res.retorno, axis=1) # para evaluar segun sharpe ratio
#    print("sharpe shape:",sharpe.shape,sharpe[0],sharpe[1])
    idx = np.argsort(sharpe)[::-1][:cnf.nsel]
    res3.reorder(idx) # ordeno todo los resultados segun el sharpe
    dias_positivos = (res.retorno > 0).sum(axis=1) # para evaluar segun cantidad de dias positivos
    idx = np.argsort(dias_positivos)[::-1][:cnf.nsel]
    res4.reorder(idx) # ordeno los resultados por los de mas dias positivos
    caps[0].append(res.retorno[:5,cnf.Ntraining:].mean(0))
    caps[1].append(res.retorno[:10,cnf.Ntraining:].mean(0))
    caps[2].append(res.retorno[:20,cnf.Ntraining:].mean(0))
    caps[3].append(res2.retorno[:5,cnf.Ntraining:].mean(0))
    caps[4].append(res2.retorno[:10,cnf.Ntraining:].mean(0))
    caps[5].append(res2.retorno[:20,cnf.Ntraining:].mean(0))

#    caps[0].append(res3.retorno[:5,cnf.Ntraining:].mean(0))
#    caps[1].append(res3.retorno[:10,cnf.Ntraining:].mean(0))
#    caps[2].append(res3.retorno[:20,cnf.Ntraining:].mean(0))
#    caps[3].append(res4.retorno[:5,cnf.Ntraining:].mean(0))
#    caps[4].append(res4.retorno[:10,cnf.Ntraining:].mean(0))
#    caps[5].append(res4.retorno[:20,cnf.Ntraining:].mean(0))


rets = np.array([np.concatenate(cap) for cap in caps])
print(rets.shape)
caps=np.zeros_like(rets)
print(caps.shape)
caps[:,0]=100
for i in range(6):
    caps[i,1:] = caps[i,0] * np.cumprod(1 + rets[i,1:])

    
figfile=cnf.fname+'marine.png'
fig, ax = plt.subplots(1,1,figsize=(7,5))
ax.plot(caps[0],label='half life 5')
ax.plot(caps[1],label='half life 10')
ax.plot(caps[2],label='half life 20')
#ax.plot(caps[3],'--C0',label='pval 5')
ax.plot(caps[4],'--C1',label='pval 10')
ax.plot(caps[5],'--C2',label='pval 20')
ax.set(title='pval testing')
ax.legend()
plt.tight_layout()
fig.savefig(figfile)
plt.close()


quit()
#for cap in res.capital[:,-1]:
#    print(cap)
#
#for company_pair in res.company:
#    print(company_pair)

metrics = co.stats(res.assets,'log')


figfile=cnf.fname+'capital.png'
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
