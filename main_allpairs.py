''' Pair trading  con todos los pares
       z-scores 
            Choice of beta calculacion: regresion / kalman filter
            Choice of averaging:
                  Using moving average window / exponential mean averaging / kalman filter      
'''
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

class cnf:
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'on'# 'kf' 'exp' 'on' 'off'
    Ntraining = 2*252 # length of the training period
    Npred=Ntraining+252
    beta_win=121   #21
    zscore_win=41 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
    nmax=-1 # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=100# 100 # number of best pairs to select
    fname=f'tmp/all_pair_{mtd}_' # fig filename
    linver_betaweight=0
    #industry='oil'
    industry='beverages'
    shorten=0
    
# load data
day,date,price,company,volume = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)


# select training period
assets_tr=price[:cnf.nmax,:cnf.Npred]

t0 = time()
res = ar.all_pairs(assets_tr,company[:cnf.nmax],cnf)
print('Tiempo:  ',time()-t0)

res2=copy.deepcopy(res)

# Select nsel best pairs
idx = np.argsort(res.capital[:,cnf.Ntraining])[::-1][:cnf.nsel]
res.reorder(idx) # ordeno todo los resultados segun el capital
cap_pred=utils.returns_from(res.capital,cnf.Ntraining)


metrics = co.all_pairs_stats(assets_tr[:,:cnf.Ntraining],company,'asset')
idx = np.argsort(metrics.pvalue)[:cnf.nsel]
res2.reorder(idx) # ordeno todo los resultados segun el p-value
cap_pred2=utils.returns_from(res2.capital,cnf.Ntraining)

figfile=cnf.fname+'capital.png'
fig, ax = plt.subplots(3,1,figsize=(7,7))
ax[0].plot(res.capital[:10,:].mean(0))
ax[0].plot(res.capital[:20,:].mean(0))
ax[0].plot(res.capital[:40,:].mean(0))
ax[0].set(title='capital')
ax[1].plot(cap_pred[:10,:].mean(0))
ax[1].plot(cap_pred[:20,:].mean(0))
ax[1].plot(cap_pred[:40,:].mean(0))
ax[1].plot(cap_pred2[:10,:].mean(0),'--C0')
ax[1].plot(cap_pred2[:20,:].mean(0),'--C1')
ax[1].plot(cap_pred2[:40,:].mean(0),'--C2')
ax[1].set(title='capital pred')
ax[2].plot(res2.capital[:10,:].mean(0))
ax[2].plot(res2.capital[:20,:].mean(0))
ax[2].plot(res2.capital[:40,:].mean(0))
ax[2].set(title='capital pvalue')
plt.tight_layout()
fig.savefig(figfile)
plt.close()

#for cap in res.capital[:,-1]:
#    print(cap)
#
for company_pair in res.company:
    print(company_pair)
quit()

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
