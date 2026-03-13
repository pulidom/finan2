''' Ejemplo de main para Pair trading 
       z-scores using moving average window 
'''
import numpy as np, os, copy
import matplotlib.pyplot as plt
import copy
from time import time
import datetime
from itertools import permutations

from read_data import load_ts
from cointegracion import zscore_moving_win,invierte
import seleccion 
import utils

class cnf:
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    Ntraining = 3*252 # length of the training period
    beta_win=121   #21
    zscore_win=41 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
    nmax=None # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=10#  number of best pairs to select
    industry=['oil']#['beverages']
    fname=f'tmp/all_pair_oil_' # fig filename
    
# load data
day,date,price,company,volume = load_ts(sectors=cnf.industry, pathdat=cnf.pathdat)

#metrics = seleccion.all_pairs_stats(price[:,:cnf.Ntraining],company,'asset')
#idx_selected_pairs = np.argsort(metrics.pvalue)[:cnf.nsel]

idx_selected_pairs = [3298, 3594, 1090, 3279, 1125,  404,  951,  782, 2393, 1731] # selected pairs by pvalue
indices_permutations = list(permutations(range(price.shape[0]), 2))
idx_selected_assets = [indices_permutations[i] for i in idx_selected_pairs]

print(idx_selected_assets)

price_l = [  price[idx_selected_assets[i][0],Ntraining:],
                  price[idx_selected_assets[i][1],Ntraining:]
                  for i in idx_selected_assets ]
company_l = [  company[idx_selected_assets[i][0]],
             company[idx_selected_assets[i][1]]
             for i in idx_selected_assets ]

def inversion(x,y,cnf):
    ' Pair trading para un par de assets '

    x,y,nret_x,nret_y = utils.select_variables(x,y,tipo=cnf.tipo)
    zscore0 = zscore_moving_win(x, y, beta_win=cnf.beta_win, zscore_win=cnf.zscore_win)
    compras0,ccompras0 = invierte( zscore0, cnf.sigma_co, cnf.sigma_ve )
    
    return x,y,zscore0,compras0,ccompras0

