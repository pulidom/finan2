''' Pair trading  con todos los pares
    utilizando volumen para pesar los retornos
'''
import numpy as np, os, copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from time import time

from read_data import load_ts,load_by_tickers

import arbitrage as ar
import plotting as gra
import cointegration as co
import utils

#plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')


class cnf:
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'on'# 'kf' 'exp' 'on' 'off'
    #mtd='kf'
    Ntraining = 131 # length of the training period
    Njump = 70
    beta_win=121   #21
    zscore_win=41 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.2 # thresold to sell
    nmax=-1 # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=10# 100 # number of best pairs to select
    fname=f'tmp/reu_05_08/' # fig filename
    linver_betaweight=0
    industry='software'
    shorten=0

os.makedirs(cnf.fname, exist_ok=True)    

day,date,price,company,volume = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)

#tickers = ['GOOG.O', 'GOOGL.O', 'AMZN.O', 'AAPL.O', 'AVGO.O', 'META.O', 'MSFT.O', 'NVDA.O', 'TSLA.O']
#day, date, price, company, volume = load_by_tickers(tickers)

caps = [[] for _ in range(6)]   # Para guardar resultados de las 6 estrategias
price=price
nt=price.shape[1]
iini=0

for ilast in range(cnf.Ntraining+cnf.Njump,nt,cnf.Njump):
    #print(ilast)
    assets_tr = price[:cnf.nmax,iini:ilast]
    volume_tr = volume[:cnf.nmax, iini:ilast]
    mean_vol  = np.mean(volume_tr, axis=1)  # promedio por activo    
 
    t0 = time()
    res_ = ar.all_pairs(assets_tr,company[:cnf.nmax],cnf)
    res_list = [copy.deepcopy(res_) for _ in range(6)]


    ret_segun_volumen = np.zeros(res_.retorno.shape)
    ret_segun_volumen_2 = np.zeros(res_.retorno.shape)
    w_volatil = np.zeros(res_.retorno.shape)
    w_volatil_2 = np.zeros(res_.retorno.shape)
    ret_segun_volatil_2 = np.zeros(res_.retorno.shape)
    
    
    ### Selección de pares según P-val
    res = res_list[0]
    metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],res.company[:cnf.nmax],'asset')
    idx = np.argsort(metrics.pvalue)[:cnf.nsel]
    res.reorder(idx)
    res=copy.deepcopy(res)
    caps[0].append(res.retorno[:10,cnf.Ntraining:].mean(0))
    
    ### Selección de pares según P-val
    ### Inversión utilizando pesos según volumen
    res = copy.deepcopy(res_list[1])
    metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],res.company,'asset')
    idx = np.argsort(metrics.pvalue)[:cnf.nsel]
    res.reorder(idx)
    res = ar.given_pairs_weighted_(assets_tr, company, cnf, res,volume)
    caps[1].append(res.retorno_ponderado[cnf.Ntraining:])

    ### Selección de pares según P-val
    ### Inversión utilizando pesos según volatilidad
    res = copy.deepcopy(res_list[2])    
    metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],res.company,'asset')
    idx = np.argsort(metrics.pvalue)[:cnf.nsel]
    res.reorder(idx)
    res = ar.given_pairs_weighted_(assets_tr, company, cnf, res,volume,ar.volatility_weight)
    caps[2].append(res.retorno_ponderado[cnf.Ntraining:])

    ### seleccion de pares segun HALF LIFE
    res = res_list[2]
    metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],company,'asset')
    idx = np.argsort(metrics.half_life)[:cnf.nsel]
    res.reorder(idx)
    res=copy.deepcopy(res)
    caps[3].append(res.retorno[:10,cnf.Ntraining:].mean(0))
    
    #print('HL:  ',time()-t0)
    
    ### Selección de pares según HALF LIFE
    ### Inversión utilizando pesos según volumen
    res = copy.deepcopy(res_list[3])
    metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],res.company,'asset')
    idx = np.argsort(metrics.half_life)[:cnf.nsel]
    res.reorder(idx)
    res = ar.given_pairs_weighted_(assets_tr, company, cnf, res,volume)
    caps[4].append(res.retorno_ponderado[cnf.Ntraining:])

    ### Selección de pares según HALF LIFE
    ### Inversión utilizando pesos según volumen
    res = copy.deepcopy(res_list[3])
    metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],res.company,'asset')
    idx = np.argsort(metrics.half_life)[:cnf.nsel]
    res.reorder(idx)
    res = ar.given_pairs_weighted_(assets_tr, company, cnf, res,volume,ar.volatility_weight)
    caps[5].append(res.retorno_ponderado[cnf.Ntraining:])
    print('inicio:  ',time()-t0)
    iini+=cnf.Njump


rets = np.array([np.concatenate(cap) for cap in caps])
capitales=np.zeros_like(rets)
capitales[:,0]=100

for i in range(len(caps)):
    capitales[i,1:] = capitales[i,0] * np.cumprod(1 + rets[i,1:])

print('capitales 0 ',capitales[0].shape)

figfile = f'{cnf.fname}{cnf.industry}.png'
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(capitales[0], '-', color='tab:blue'  ,label='pval'             , alpha = 1)
ax.plot(capitales[1], '--',color='tab:blue'  ,label='pval_volume_W'    , alpha = 0.7)
ax.plot(capitales[2], ':', color='tab:blue'  ,label='pval_volatility_W', alpha = 0.7)
ax.plot(capitales[3], '-', color='tab:orange',label='HL'               , alpha = 1)
ax.plot(capitales[4], '--',color='tab:orange',label='HL_volume_W'      , alpha = 0.7)
ax.plot(capitales[5], ':', color='tab:orange',label='HL_volatility_W'  , alpha = 0.7)

ax.set(title=f'Capitales en {cnf.industry}')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(figfile)
plt.close()