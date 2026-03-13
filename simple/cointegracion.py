import numpy as np, os
import utils
from utils import (lin_reg, mean_function,)


def zscore_moving_win(x, y, beta_win=41, zscore_win=21, eps = 1.e-10 ):
    ''' Usa regresion lineal y una ventana movil para calcular el z_score 
       '''    
    beta = np.full_like(x, np.nan)
    spread = np.full_like(x, np.nan)
    spread_mean = np.full_like(x, np.nan)
    spread_std = np.full_like(x, np.nan)
    spread_sq = np.full_like(x, np.nan)
    zscore = np.full_like(x, np.nan)
    
    for it in range(beta_win, len(x)):
        x_win = x[it-beta_win:it]
        y_win = y[it-beta_win:it]

        beta[it],alpha = lin_reg(x_win, y_win)

        spread[it] = y[it] - (beta[it] * x[it] + alpha)

        if it >= zscore_win:
            spread_win = spread[it-zscore_win+1:it+1] # incluye el it
            spread_mean[it],spread_std[it] = utils.meanvar(spread_win)
            zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it]+eps)
            
    return zscore,beta,spread,spread_mean,spread_std

def inversion(x,y,cnf):
    ' Hago todo el proceso  para un par de assets '

    x,y,nret_x,nret_y = utils.select_variables(x,y,tipo=cnf.tipo)
    zscore0,b,s,sm,ss =  zscore_moving_win(x, y, beta_win=cnf.beta_win,
                                           zscore_win=cnf.zscore_win)
    compras0,ccompras0 = invierte( zscore0, cnf.sigma_co, cnf.sigma_ve )

    beta = b if cnf.linver_betaweight else None
    largo0, corto0, capital0,retorno0 = capital_invertido(nret_x,nret_y,
                                                 compras0,ccompras0,
                                                 beta=beta)
    res={  'asset_x': x,'asset_y': y,'largo':largo0, 'corto':corto0,
           'capital':capital0, 'retorno':retorno0,
           'compras':compras0, 'ccompras':ccompras0, 'zscore':zscore0,
           'beta':b, 'spread':s, 'spread_mean':sm, 'spread_std':ss }
    return res

def invierte(zscore,sigma_co=1.5,sigma_ve=0.5):
    ''' Determina los intervalos temporales de compra venta en una serie
        single time series
       '''

    compras=np.zeros(zscore.shape[0], dtype=bool)
    ccompras=np.zeros(zscore.shape[0], dtype=bool)
    band,cband=0,0
    for it in range(zscore.shape[0]):
        if band: # poseo el activo
            if zscore[it] > sigma_ve:
                compras[it]=True # mantengo
            else:
                band=0 # vendo
        else: # no poseo el activo
            if zscore[it] > sigma_co:
                band=1 # compro
                compras[it]=True
        # posiciones en corto
        if cband:
            if zscore[it] < - sigma_ve:
                ccompras[it]=True # mantengo
            else:
                cband=0 # vendo
        else:
            if zscore[it] < - sigma_co:
                cband=1 # compro
                ccompras[it]=True
    return compras,ccompras

def capital_invertido(nret_x,nret_y,compras,ccompras,beta=None):
    ''' invierto el capital con pares
        divide la inversion en forma equitativa o con beta weights
          compras z_score > 0 y ccompras z_score < 0
        nret_x= (x[it+1]-x[it])/x[it] (normalized return)
     '''
    
    corto, largo = np.zeros((2,nret_x.shape[0],2))
    capital,retorno = np.zeros(( 2,nret_x.shape[0] ))
    largo[0] = 100
    corto[0] = 100
    capital[0] = 100
    for it  in range(nret_x.shape[0]-1):
        
        if beta is None:
            w_x=w_y=1
        else:
            w_x=1/(1+np.abs(beta[it]))
            w_y=np.abs(beta[it])/(1+np.abs(beta[it]))
        
        if compras[it] > 0:
            retorno[it+1] = 0.5*(w_x * nret_x[it]-w_y *nret_y[it])
            capital[it+1] = capital[it] * (1+retorno[it+1])
            largo[it+1,0] = largo[it,0] * (1+w_x*nret_x[it]) 
            corto[it+1,1] = corto[it,1] * (1-w_y*nret_y[it])
        else:
            largo[it+1,0] = largo[it,0]
            corto[it+1,1] = corto[it,1]
        if ccompras[it] > 0:
            retorno[it+1] = 0.5*(-w_x*nret_x[it]+w_y*nret_y[it])
            largo[it+1,1] = largo[it,1] * (1+w_y*nret_y[it]) 
            corto[it+1,0] = corto[it,0] * (1-w_x*nret_x[it])
            capital[it+1] = capital[it] * (1+retorno[it+1])
        else:
            largo[it+1,1] = largo[it,1]
            corto[it+1,0] = corto[it,0]
        if compras[it] == 0 and ccompras[it] == 0:
            capital[it+1] = capital[it]
        #capital=largo.sum(1)+corto.sum(1)
    return largo, corto,capital,retorno
