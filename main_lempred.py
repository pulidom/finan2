""" Linear filter with constant parameters (regressed in the EM)
"""
import sys;  sys.path.insert(0, '../');  sys.path.insert(0, './')
import numpy as np, os
import matplotlib.pyplot as plt
import numpy.random as rnd
from enkf.ldyn import ldyn
import enkf.obs as obs
import enkf.utils_enkf as utils
from read_data import load_ts
import stats
#from inversiones import (invierte,
#                         capital_invertido,
#                         calculate_metrics,
#                         capital_invertido_var,
#                         calc_startend)
#from enkf import FILTER
from enkf.em import FILTER, LFILTER, exp_iniem, init_spar
from plotting import plot_prediccion, plot_asset, plot_std

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
    nmax=None # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=100# 100 # number of best pairs to select
    linver_betaweight=0
    industry=['beverages'] #['oil'] # ['beverages']
    fname=f'tmp/all_pair_coca-pepsi2pairs_{mtd}_' # fig filename
    #assets=['KGEI.O','CNX']
    assets=['KO','PEP.O']
    #assets=None
    # TRGP.K-GPRE.O
    # KGEI.O-CNX
    # KO-PEP
    shorten=0

class cnff:
    dtcy=1.
    niter_first=20 #20 # 20 #40
    niter_cont=5 # 5 #5
    
    Nem=1000
    fname='tmp/linear_5var_em-enkf_'
    llast=0

    Nvar = 2
    Ny   = 2
    Nx   = Nvar*Nvar+Nvar
    indobs = np.arange(Ny)
    Ncy  = 200
    sigma_y = 1
    var0obs = 0.25
    rel_error=0.05
    Q0   = [0.1,0.0] # 0.01,0.00001]
    # inversion
    Npred=30
    sigma_co=1.0
    sigma_ve=0.0
    
    finf=np.ones(Nx)#*1.05
    
class o_kwargs:
    H = obs.obsind2H(cnf.indobs,cnf.Nx)
    R = cnf.sigma_y * np.eye(cnf.Ny) 
    swH=0

class o_kwargs1v: # una sola variable observada (el otro asset)
    H = obs.obsind2H([1],cnf.Nx)
    R = cnf.sigma_y * np.eye(1) 
    swH=0
        
# load data
day,date,price,company,volume = load_ts(sectors=cnf.industry, assets=cnf.assets, 
                                        pathdat=cnf.pathdat)

quit()
y_full = price[:-1,:cnf.Ny]
Nt = int((day.shape[0]-cnf.Ncy)/cnf.Npred) * cnf.Npred 

# first EM
y_t = y_full[:cnf.Ncy]
x0,var0,Q0 = init_spar(y_t[0],cnf)
var0[cnf.Nx:]=0.0 # no spread in the parameters

res,rmse, Filter = exp_iniem(o_kwargs,cnf,y_t=y_t,
                             x0=x0,var0=var0,Q0=Q0,
                             sfilter="lfilter"
                             )

inver= np.zeros((Nt,cnf.Ny)) # cuantas variables
lastcompra, lastccompra=np.zeros((2,cnf.Ny), dtype=bool)
Filter.niter = cnf.niter_cont
Xa,Xf =  np.zeros([2,Nt,cnf.Nx,cnf.Nem])
compras, ccompras=np.zeros((2,Nt,cnf.Ny), dtype=bool)

for it  in range(cnf.Ncy, Nt+cnf.Ncy, cnf.Npred):
    
    # Filter for predictions
    X0=np.zeros((cnf.Nx,cnf.Nem))
    #X0=res[1]['Xa'][-1]+P_s[:cnf.Nvar,:cnf.Nvar] * rndn(cnf.Nx,cnf.Nem) # la Q se actualiza sola
    print(res[1]['Xa'][-1][cnf.Nvar:])
    X0[:cnf.Nvar,:] = rnd.multivariate_normal(res[1]['Xa'][-1][:cnf.Nvar],res[1]['Ps'][-1][:cnf.Nvar,:cnf.Nvar],cnf.Nem).T
    X0[cnf.Nvar:,:] = res[1]['M'][:cnf.Nvar,:cnf.Nvar].reshape(cnf.Nx-cnf.Nvar)[:,None]
    print(Filter.sqQ[0,0])
    X0 = Filter.integ_noise(X0)
    y_t = y_full[it:it+cnf.Npred]
    Xa_t,Xf_t = Filter.filter(X0,y_t) # hago predicciones y analisis

    it0=it-cnf.Ncy
    it1=it0+cnf.Npred
    Xa[it0:it1]=Xa_t
    Xf[it0:it1]=Xf_t

    compras[it0:it1],ccompras[it0:it1]=invierte(Xf_t[:,:cnf.Ny],y_t,
                                                compras[it0-1],ccompras[it0-1],
                                                sigma_co=cnf.sigma_co,
                                                sigma_ve =cnf.sigma_ve)    
    # Calculate EM 
    it0=it-cnf.Ncy+cnf.Npred
    y_t = y_full[it0:it0+cnf.Ncy]
    res,rmse=Filter(X0,y_t)



# calculo los returnscompras = np.concatenate(compras)
returns = ((price[cnf.Ncy+1:Nt+cnf.Ncy+1,:cnf.Ny]-price[cnf.Ncy:Nt+cnf.Ncy,:cnf.Ny])
           /price[cnf.Ncy:Nt+cnf.Ncy,:cnf.Ny])
y=y_full[:Nt]

# corrido hacia adelante (compro un dia y veo ganancia al dia siguiente).
# Aca hay una cuestion de cuando compro. En realidad si ya tengo el precio de cierre
# recien al otro dia podria comprar
returns_comprado  =   returns * compras
returns_ccomprado = - returns * ccompras
capital, capital_largo, capital_corto = capital_invertido(y,compras,ccompras)
spread =  utils.spread(Xf[:,:cnf.Ny])[:-1]

#----------------------------------------------------
start_date = datetime.datetime(2020, 1, 1)
day=day[cnf.Ncy:y.shape[0]+cnf.Ncy]
dates = np.array([start_date + datetime.timedelta(days=int(d)) for d in day])

nini=0
nfin=Nt-2

plot_prediccion( 50,100,compras,ccompras,y,Xf,cnf.sigma_co,
                  figfile=cnf.fname) #,dates=dates)
plot_prediccion(100,150,compras,ccompras,y,Xf,cnf.sigma_co,
                figfile=cnf.fname)
cap=np.stack([capital,capital_largo,capital_corto])
plot_std(nini,nfin,[returns_comprado.T,returns_ccomprado.T,cap],
         labels=['Retorno largo','Retorno corto',['Capital','Largo','Corto']],
         dates=dates,figfile=cnf.fname+'_capital')
plot_std(100,150,[returns_comprado.T,returns_ccomprado.T,cap],
         labels=['Retorno largo','Retorno corto',['Capital','Largo','Corto']],
         dates=dates,figfile=cnf.fname+'_capital')
plot_std(nini,nfin,spread.T,dates=dates,figfile=cnf.fname+'_spread')
plot_asset(50,100,compras,ccompras,y,
           figfile=cnf.fname)
plot_asset(100,150,compras,ccompras,y,
           figfile=cnf.fname)


#nini=400
#nfin=450
#jdias=np.arange(nini,nfin)
#X = Xa.mean(-1)[nini:nfin]
#for i in range(20):
#    figfile=f'tmp/adapt_em-enkf_pars{i}.png'
#    fig, ax = plt.subplots(1,1,figsize=(4,3))
#    ax.plot(jdias,X[:,i])
#    fig.savefig(figfile)
#    plt.close()
#
