'''
Testeo de distintas métricas para selección de pares en todos los sectores a la vez.

Opera como main_operativo, es decir:
    mira las métricas durante Ntraining días
    invierte durante Njump días
    para la inversión se asignan pesos en función de la métrica de selección
        (esto no lo hacía main_operativo)
    repite

Notas:
    Sigue implementada la forsoza selección de 5-10-20 pares
    No está curado el Johansen
    La asignación de pesos se hace utilizando como parámetro la misma métrica
        que se utilizó para la selección de pares, no el rendimiento, no la
        volatilidad, no el volumen. Es decir que si la selección se hizo
        utilizando p_values, el peso se asigna con p_values.
    Cuando se selecciona según half_life se está seleccionando los pares con
        MENOR half life. Para que se seleccione el half_life dentro de cierto
        intervalo debería usarse el half_life_penalty (implementado dentro de
        cointegration_score).
    
    

Resultado:
    HF despunta de sobremanera al seleccionar los pares con MENOR vida media.
    Creo que al asignarle peso según HF de cierto modo le estamos metiendo un
    input sobre la volatilidad. No estoy seguro de si es bueno o malo.


@ContardiG
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from time import time
from collections import Counter

from read_data import load_all_ts
import arbitrage as ar
import cointegration as co

name = os.path.basename()
os.makedirs('tmp/'+name, mode=0o777, exist_ok=False)

class cnf:
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'on'# 'kf' 'exp' 'on' 'off'
    Ntraining = 126 # length of the training period
    Njump = 21
    beta_win=121   #21
    zscore_win=41 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
    nmax=200 # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=20# 100 # number of best pairs to select
    fname=f'tmp/{name}/metrics' # fig filename
    linver_betaweight=0
    shorten=0
    random_seed=42
    
# load data
day,date,price,company = load_all_ts(pathdat=cnf.pathdat)

### acá selecciono un maximo de empresas al azar porque matchear TODAS es caro
if cnf.nmax > 0 and cnf.nmax < len(company):
    np.random.seed(cnf.random_seed)
    idx_sample = np.random.choice(len(company), cnf.nmax, replace=False)
    price = price[idx_sample]
    company = company[idx_sample]

caps = [[] for _ in range(15)]  

nt=price.shape[1]
iini=0

# select training period
for ilast in range(cnf.Ntraining+cnf.Njump,nt,cnf.Njump):
    
    assets_tr=price[:cnf.nmax,iini:ilast]
    iini+=cnf.Njump
    
    t0 = time()
    res = ar.all_pairs(assets_tr,company[:cnf.nmax],cnf)
    print('Tiempo:  ',time()-t0)

    # Select nsel best pairs
    idx = [np.argsort(res.capital[:,ilast-cnf.Njump-iini])[::-1][:cnf.nsel]]
    metrics = co.all_pairs_stats(assets_tr[:,:ilast-cnf.Njump],company,'asset')
    
    metric_values = {'pvalue': np.array(metrics.pvalue),'johansen': np.array(metrics.johansen),'hurst': np.array(metrics.hurst),'half_life': np.array(metrics.half_life)}

    res_list = [copy.deepcopy(res) for _ in range(4)]
    res_list.insert(0,res)

    idx_names = ['pvalue', 'johansen', 'hurst', 'half_life']
    idx_list = [np.argsort(getattr(metrics, name))[:cnf.nsel] for name in idx_names]
    idx_list = idx+idx_list
    
    for i in range(5):
        res_list[i].reorder(idx_list[i])

    invert_metric = [False, False, False, False, False]  # capital, pval, joh, hurst, half-life
    #invert_metric = [False, True, False, True, True]     # capital, pval, joh, hurst, half-life
    

    # Promedios ponderados para 5, 10, 20 pares seleccionados
    for i in range(5):  # para cada métrica de selección
        metric_name = 'capital' if i == 0 else idx_names[i - 1]
        metric_array = (res.capital[:, ilast - cnf.Njump - iini] if i == 0 else metric_values[metric_name])
        
        ### Si se quiere usar el capital como métrica: descomentar la sig linea
        #metric_array = res.capital[:, ilast - cnf.Njump - iini]
        
        for j, n in enumerate([5, 10, 20]):

            values = metric_array[:n]

            if invert_metric[i]:
                values = 1 / (np.array(values) + 1e-8)  # para evitar división por cero

            weights = co.compute_weights(values)
            weighted_ret = np.average(res_list[i].retorno[:n, cnf.Ntraining:], axis=0, weights=weights)
            caps[3 * i + j].append(weighted_ret)
            
            #top_pairs = res_list[i].assets[:n]  # lista de tuplas (a1, a2)
            top_pairs = [tuple(map(str, pair)) for pair in res_list[i].assets[:n]]
            pair_counter.update(top_pairs)

rets = np.array([np.concatenate(cap) for cap in caps])
print(rets.shape)

caps=np.zeros_like(rets)
print(caps.shape)
caps[:,0]=100

for i in range(15):
    caps[i,1:] = caps[i,0] * np.cumprod(1 + rets[i,1:])



### plotting

fig, ax = plt.subplots(3,1,figsize=(7,7),dpi=300,layout='tight')
met_lab=['cap','pval', 'joh', 'hur', 'hf', 'sc']

for i in range(5):
    ax[0].plot(caps[i*3],label=met_lab[i]+'_5')
    ax[1].plot(caps[i*3+1],label=met_lab[i]+'_10')
    ax[2].plot(caps[i*3+2],label=met_lab[i]+'_20')

ax[0].legend(loc='upper left');ax[1].legend(loc='upper left');ax[2].legend(loc='upper left')
ax[0].grid();ax[1].grid();ax[2].grid()
fig.suptitle(f'Evolución del capital para cada métrica de selección en todos los sectores')

for ax in ax[:-1]:
    ax.label_outer()  

fig.savefig(cnf.fname+'_capital')

quit()
