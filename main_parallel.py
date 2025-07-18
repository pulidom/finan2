''' Optimizacion de hyperparameters de pair trading
      Con multiprocessing. Para correr en servidores
     La idea es seleccionar nsel=50 pares con los parametros optimos para cada par
     y luego quedarnos con los 10 mejores pares 
'''
import multiprocess
import numpy as np, os, psutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.random as rnd
import datetime
import itertools
import pickle
from read_data import load_ts
import arbitrage as ar
#import hypero as ho  
#from signal import signal, SIGPIPE, SIG_DFL  # avoid broken pipe paralell computing
#signal(SIGPIPE,SIG_DFL)
class cnf:
    ncores = psutil.cpu_count(logical = False)
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'kf' # 'kf' 'exp' 'on' 'off'
    industry='oil'
    filedat=f'hypero_{tipo}_{mtd}_'
    nmax=-1 # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=100# 100 # number of best pairs to select
    linver_betaweight=1 #
    lsharpe=0

    # hyperparametros a optimizar
    # 
    #    beta_win=#[121,111,91,71,51,31]
    #    zscore_win=#[121,101,81,61,41,21]
    beta_win=121 # to be overwritten
    zscore_win=63
    Ntraining_l = [63, 126, 189, 252, 378, 504]
    Njump_l = [21,42,63,84,105,126]
    sigma_co_l=[2.0,1.8,1.6,1.4]
    sigma_ve_l=[0.0,0.15,0.3,0.45]
    
# load data
day,date,price,company,volume = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)

# Genero las combinaciones de todos los hyperparametros
params = [list(comb) for comb in itertools.product(
    cnf.Ntraining_l, cnf.Njump_l, cnf.sigma_co_l, cnf.sigma_ve_l)]

class All_pairs:
    '''  Clase para tener un solo argumento para multiprocess'''
    def  __init__(self,price,company,cnf):
        self.price = price
        self.company = company
        self.cnf = cnf
        self.nt=price.shape[-1]
    def fn(self,arg):
        Ntraining = arg[0]
        Njump = arg[1]
        self.cnf.sigma_co = arg[2]
        self.cnf.sigma_ve = arg[3]
        self.cnf.beta_win=Ntraining
        iini=0
        for ilast in range(Ntraining+Njump,self.nt,Njump):
            print(iini,ilast,ilast-iini)

            assets_tr=self.price[:self.cnf.nmax,iini:ilast]

            print(assets_tr.shape)
            iini+=Njump

            res = ar.all_pairs(assets_tr,company[:self.cnf.nmax],self.cnf)

            # Select nsel best pairs
            idx = np.argsort(res.capital[:,ilast-Njump-iini])[::-1][:self.cnf.nsel]
            res.reorder(idx) # ordeno todo los resultados 
            
        return res.retorno
    __call__=fn

pairs = All_pairs(price,company,cnf)
res=[]
# reparto procesos en los cores del cpu
with multiprocess.Pool(cnf.ncores) as p:               
     res.append( p.map(pairs, params) )
res = [ent for sublist in res for ent in sublist]

# Guardo resultados 
with open(cnf.pathdat+cnf.filedat+'res.pkl', 'wb') as f: 
    pickle.dump(res, f)
