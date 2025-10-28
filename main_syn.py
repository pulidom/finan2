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
import plotting
#from plotting import vertical_bar

class cnf_cls:
    pathdat='dat/'
#    tipo='asset' # 'asset', 'return', 'log_return', 'log'
#    mtd = 'on'# 'kf' 'exp' 'on' 'off'
    tipo='log' #asset'#log' #'asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'ot2'# 'kf' 'exp' 'on' 'off'
    Njump = 84
    beta_win=121 #*121   #21
    zscore_win=41 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
#    nmax=None # number of companies to generate the pairs (-1 all, 10 for testing)
#    nsel=100# 100 # number of best pairs to select
    linver_betaweight=0
#    industry=['beverages'] #['oil'] # ['beverages']
    fname=f'tmp/syn_pair_{mtd}_' # fig filename
    shorten=0

cnf_co = cnf_cls(); cnf = cnf_cls()
cnf_co= copy.deepcopy(cnf)
cnf_co.tipo='asset'
cnf_co.mtd='on'

# load data
nt= 10*252
ts = load_sts(nt=nt,lopt=2,regime_length=nt,seed=43)#252)

ts=ts[:2]


print('Nro de tiempos/dias',nt)
iini=0
res_l=[]; res_co_l=[]
# select training period
for ilast in range(cnf.beta_win+cnf.Njump,nt,cnf.Njump):
    print(iini,ilast,ilast-iini)
    
    x,y=ts[:,iini:ilast]
    iini+=cnf.Njump

    t0 = time()
    res = ar.inversion(x,y,cnf,shorten=cnf.shorten)
    res['asset_x']=x
    res['asset_y']=y
    res_l.append(res)

    res_co = ar.inversion(x,y,cnf_co,shorten=cnf.shorten)
    res_co_l.append(res_co)


       
#ret1 = np.concatenate([res['retorno'][cnf.beta_win:] for res in res_l],axis=0)
res = {
    key: np.concatenate([res[key][cnf.beta_win:] for res in res_l],axis=0)
    for key in res_l[0].keys()  # Usa las keys del primer diccionario
    }
res_co = {
    key: np.concatenate([res[key][cnf.beta_win:] for res in res_co_l],axis=0)
    for key in res_co_l[0].keys()  # Usa las keys del primer diccionario
    }

#for key in res_l[0].keys():  # Usa las keys del primer diccionario
#    print(key)

res = utils.dict2obj(**res) # mas elegante con objetos! :)
res_co = utils.dict2obj(**res_co) # mas elegante con objetos! :)

res.capital=np.zeros_like(res.retorno)
res.capital[0]=100
res.capital= res.capital[0] * np.cumprod(1 + res.retorno)

res_co.capital=np.zeros_like(res_co.retorno)
res_co.capital[0]=100
res_co.capital= res_co.capital[0] * np.cumprod(1 + res_co.retorno)

#res.asset_x=x
#res.asset_y=y

t=np.arange(res.asset_x.shape[0])/252
   
figfile=cnf.fname+'scatter.png'
fig, ax = plt.subplots(1,1,figsize=(7,5))
scatter = plt.scatter(res.asset_x, res.asset_y, c=t, cmap='viridis', s=10, alpha=0.7)
plt.colorbar(scatter, label='Time')

#ax.plot(res.asset_x,res.asset_y,'.',label='cap 5')
#ax.plot(res.capital[0],label='cap 5')
#ax.plot(res.capital[1],label='cap 10')
ax.set(title='Scatterplot activos')
plt.tight_layout()
fig.savefig(figfile)
plt.close()
   
t=np.arange(res.capital.shape[0])/252
   
figfile=cnf.fname+'capital.png'
fig, ax = plt.subplots(1,1,figsize=(7,5))
ax.plot(t,res.capital,label='Tabak')
ax.plot(t,res_co.capital,label='Cointegration')
#ax.plot(res.capital[0],label='cap 5')
#ax.plot(res.capital[1],label='cap 10')
ax.set(title='capital testing')
ax.legend()
plt.tight_layout()
fig.savefig(figfile)
plt.close()


t=np.arange(res.zscore.shape[0])/252
figfile=cnf.fname+'zscores.png'
fig, ax = plt.subplots(2,1,figsize=(7,7))
ax[0].plot(t,res.zscore)
plotting.vertical_bar([ax[0]],res.compras,res.ccompras)
ax[1].plot(t,res_co.zscore)
plotting.vertical_bar([ax[1]],res_co.compras,res_co.ccompras)
plt.tight_layout()
fig.savefig(figfile)
plt.close()




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
