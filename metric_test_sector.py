'''
Testeo de distintas métricas para selección de pares en distintos sectores.
Opera como main_operativo, es decir:
    mira las métricas durante Ntraining días
    invierte durante Njump días
    repite

Sigue implementada la forsoza selección de 5-10-20 pares
No está arreglado el Johansen (creo que multiplicándolo por -1 se arreglaría)
No está implementada la función que asigna pesos según performance

Resultado:  La vida media parece ser mejor salvo ciertos casos que compite con pvalues
            A veces el Hurst se mete en la disputa pero no mucho más

Los mejores resultados se ven en:
    biotechnology!
    machinery
    marine!
    media!
    metals
    oil

Parece haber problemas con:
    hotels

@ContardiG
'''

import numpy as np, os, copy
import matplotlib.pyplot as plt
import copy
from time import time

from read_data import load_ts
import arbitrage as ar
import cointegration as co

name = os.path.basename()
os.makedirs('tmp/'+name, mode=0o777, exist_ok=False)

industrias=["insurance","marine","metals","oil","semiconductors","software","pharmaceuticals","chemicals","capital","containers","water","machinery","beverages","media"]

for industria in industrias:
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
        nmax=-1 # number of companies to generate the pairs (-1 all, 10 for testing)
        nsel=20# 100 # number of best pairs to select
        fname=f'tmp/{name}/metrics_in_{industria}' # fig filename
        linver_betaweight=0
        #industry='oil'
        industry=industria
        shorten=0
        
    # load data
    day,date,price,company = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)
    
    ### límite de días que quiero iterar
    #lim=cnf.Ntraining+2*252    ### 2 años...
    lim=cnf.Ntraining+20*252    ### 20 es más que lo que tiene el dataset
                                ### así que iría hasta el final (10 años aprox)
    
    caps = [[] for _ in range(15)]  
    
    nt=price.shape[1]
    iini=0
    
    # select training period
    for ilast in range(cnf.Ntraining+cnf.Njump,nt,cnf.Njump):
        if ilast<lim:
            assets_tr=price[:cnf.nmax,iini:ilast]
            iini+=cnf.Njump
        
            t0 = time()
            res = ar.all_pairs(assets_tr,company[:cnf.nmax],cnf)
            print('Tiempo:  ',time()-t0)
            
            idx = [np.argsort(res.capital[:,ilast-cnf.Njump-iini])[::-1][:cnf.nsel]] ###usa como "métrica" el capital positivo
            metrics = co.all_pairs_stats(assets_tr[:,:ilast-cnf.Njump],company,'asset')
    
            res_list = [copy.deepcopy(res) for _ in range(4)]
            res_list.insert(0,res)
    
            idx_names = ['pvalue', 'johansen', 'hurst', 'half_life']
            idx_list = [np.argsort(getattr(metrics, name))[:cnf.nsel] for name in idx_names]
            idx_list = idx+idx_list
            
            for i in range(5):
                res_list[i].reorder(idx_list[i])
    
            for i in range(5):
                caps[3*i+0].append(res_list[i].retorno[:5,cnf.Ntraining:].mean(0))
                caps[3*i+1].append(res_list[i].retorno[:10,cnf.Ntraining:].mean(0))
                caps[3*i+2].append(res_list[i].retorno[:20,cnf.Ntraining:].mean(0))
    
    
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
    fig.suptitle(f'Evolución del capital para cada métrica de selección en {industria}')
    
    for ax in ax[:-1]:
        ax.label_outer()  
    
    fig.savefig(cnf.fname)

quit()
