''' Pair trading  con todos los pares
    utilizando volumen para pesar los retornos
'''
import numpy as np, os, copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from time import time
import datetime

from read_data import load_ts,load_by_tickers,yahoo_download

import arbitrage as ar
import plotting as gra
import cointegration as co
import utils
import pandas as pd

#plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')

#oil = 96 -> 61
#semiconductors = 74 -> 56
#software: 169 -> 56
#chemicals: 61 -> 40
#capital: 110 -> 65
#machinery: 98 -> 76
#industrias = ["automobiles"]
industrias = ['args']
for industria in industrias:
    
        
    #print(f'arranca {industria}')
    class cnf:
        pathdat='dat/'
        tipo='log' # 'asset', 'return', 'log_return', 'log'
        mtd = 'ot'# 'kf' 'exp' 'on' 'off'
        #mtd='kf'
        Ntraining = 252*4 # length of the training period
        Njump = 70
        beta_win=252*4   #21
        zscore_win=252*3 #11
        sigma_co=1.5 # thresold to buy
        sigma_ve=0.2 # thresold to sell
        nmax=-1 # number of companies to generate the pairs (-1 all, 10 for testing)
        nsel=10# 100 # number of best pairs to select
        fname=f'tmp/prueba_ot/' # fig filename
        linver_betaweight=0
        industry=industria

        shorten=0

    figfile = f'{cnf.fname}{cnf.industry}/plot.png'
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    os.makedirs(cnf.fname, exist_ok=True)    
    os.makedirs(cnf.fname+industria+'/', exist_ok=True)    

    #day,date,price,company,volume = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)
    #day,date,price,company,volume = yahoo_download(['AR','EQT','CNX','NAT','MTR'],datetime.date(2018,1,1),datetime.date(2022,1,1))
    day,date,price,company,volume = yahoo_download(['YPFD.BA',
                                                    'GGAL.BA',
                                                    'PAMP.BA',
                                                    'TGSU2.BA',
                                                    #'BMA-D.BA',
                                                    'TECO2.BA',
                                                    #'BBAR.BA',
                                                    #'TXAR.BA',
                                                    #'CEPU',
                                                    #'BYMA.BA',
                                                    #'LOMA',
                                                    #'EDN.BA'
                                                    ],
                                                    datetime.date(2015,1,1),datetime.date(2025,1,1))
    caps = []   # Para guardar resultados de las 6 estrategias
    nt=price.shape[1]
    
    print('descarga completada')
    #PARA GUARDAR LOS PESOS JUNTO CON LAS SEÑALES
    signal_pval = np.zeros((len(company), nt))
    signal_pval_volat = np.zeros((len(company), nt))
    signal_pval_volum = np.zeros((len(company), nt))
    signal_hf = np.zeros((len(company), nt))
    signal_hf_volat = np.zeros((len(company), nt))
    signal_hf_volum = np.zeros((len(company), nt))

    estrategias = [
    {"nombre": "pval",        "criterio": "pvalue",    "peso": None,         "signal": signal_pval      , 'line':'-'  ,'color':'tab:blue','al':1},
    #{"nombre": "pval",        "criterio": "pvalue",    "peso": None,         "signal": signal_pval      , 'line':'-'  ,'color':'red','al':1},
    {"nombre": "pval_volum",  "criterio": "pvalue",    "peso": "volume",     "signal": signal_pval_volum, 'line':'--' ,'color':'tab:blue','al':0.7},
    {"nombre": "pval_volat",  "criterio": "pvalue",    "peso": "volatility", "signal": signal_pval_volat, 'line':':'  ,'color':'tab:blue','al':0.7},
    {"nombre": "hf",          "criterio": "half_life", "peso": None,         "signal": signal_hf        , 'line':'-'  ,'color':'tab:orange','al':1},
    {"nombre": "hf_volum",    "criterio": "half_life", "peso": "volume",     "signal": signal_hf_volum  , 'line':'--' ,'color':'tab:orange','al':0.7},
    {"nombre": "hf_volat",    "criterio": "half_life", "peso": "volatility", "signal": signal_hf_volat  , 'line':':'  ,'color':'tab:orange','al':0.7},
    ]

    # Actualizar señales
    lee = len (estrategias)
    i=0
    for strat in estrategias:
        caps = []
        iini=0
        for ilast in range(cnf.Ntraining+cnf.Njump,nt,cnf.Njump):
            #print(ilast)

            assets_tr = price[:,iini:ilast]
            volume_tr = volume[:, iini:ilast]
            mean_vol  = np.mean(volume_tr, axis=1)  # promedio por activo    
        
            #t0 = time()
            res_ = ar.all_pairs(assets_tr,company,cnf,)
            res = copy.deepcopy(res_)

            ### Selección de pares según P-val
            metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],res.company,'asset')
            
            #print(len(metrics.pvalue))
            #exit()
            
            if strat['criterio']=='pvalue':
                idx = np.argsort(metrics.pvalue)[:cnf.nsel]    
            else:
                idx = np.argsort(metrics.half_life)[:cnf.nsel]
            
            if i==0:
                ar.ordenar_pares(res,volume_tr,metrics.pvalue,cnf)
            else:
                res.reorder(idx)
            #exit()
            if strat["peso"] == "volume":
                res = ar.given_pairs_weighted(assets_tr, company, cnf, res, volume_tr)
            elif strat["peso"] == "volatility":
                res = ar.given_pairs_weighted(assets_tr, company, cnf, res, volume_tr, ar.volatility_weight)

            # for p in range(len(res.company)):
            #     com0, com1 = res.company[p, 0], res.company[p, 1]
            #     pos0, pos1 = np.where(company == com0)[0][0], np.where(company == com1)[0][0]
            #     compras, ccompras = res.compras[p,-cnf.Njump:], res.ccompras[p,-cnf.Njump:]

            #     if strat["peso"]:
            #         peso_par = res.pesos_normalizados[p,-cnf.Njump:]
            #         strat["signal"][pos0, ilast-cnf.Njump:ilast] += (compras.astype(int) - ccompras.astype(int)) * np.abs(peso_par)
            #         strat["signal"][pos1, ilast-cnf.Njump:ilast] += (ccompras.astype(int) - compras.astype(int)) * np.abs(peso_par)
            #     else:
            #         strat["signal"][pos0, ilast-cnf.Njump:ilast] += compras.astype(int) - ccompras.astype(int)
            #         strat["signal"][pos1, ilast-cnf.Njump:ilast] += ccompras.astype(int) - compras.astype(int)

            # Guardar retornos
            if strat["peso"]:
                caps.append(res.retorno_ponderado[cnf.Ntraining:])
            else:
                caps.append(res.retorno[:10, cnf.Ntraining:].mean(0))

            iini+=cnf.Njump
        i=i+1
    ###aca guardo los pesos

    # Pval sin pesos
        #df = pd.DataFrame(strat['signal'].T, columns=company, index=date)
        #df.index.name = "date"
        #df.to_csv(f"{cnf.fname+industria}/{strat['nombre']}.csv")

        rets = np.concatenate(np.array(caps))
        #print(caps.shape)
        capitales=np.zeros_like(rets)
        capitales[0]=100

        capitales[1:] = capitales[0] * np.cumprod(1 + rets[1:])

        #print('capitales 0 ',capitales[0].shape)
        #print(capitales.shape)
        ax.plot(capitales, linestyle=strat['line'], color=strat['color']  ,label=strat['nombre'], alpha = strat['al'])
        fig.savefig(figfile)
        #strat['signal']=0

    ax.set(title=f'Capitales en {cnf.industry}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(figfile)
    plt.close()