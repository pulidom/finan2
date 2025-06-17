''' Compute online sequential z-scores  based on: moving averages, exponential averages and kalman filter
      beta calculation based on linear regression / kalman filter
'''
import numpy as np, os
from sklearn.linear_model import LinearRegression
from itertools import permutations
#from hurst import compute_Hc as hurste
#from statsmodels.tsa.vector_ar.vecm import coint_johansen
import utils
from utils import (rolling, erolling, crolling,
                   rolling_meanvar, exp_mean, mean_function, meanvar,
                   lin_reg,lin_reg_alpha0)


def calculate_spread( x, y,window):
    """
    Calcula el spread entre dos activos usando regresion lineal

    Usa una ventana para calcular alpha y beta y luego estima
    el spread a un tiempo
    """
    model = LinearRegression()
    spread,beta,alpha=np.full((3,y.shape[0]),np.nan)
    
    for it in range(window,x.shape[0]):
        model.fit(x[it-window:it-1,None], y[it-window:it-1])
        beta[it] = model.coef_  
        alpha[it] = model.intercept_  
        spread[it] = y[it] - model.predict([x[it,None]])

    return spread, beta, alpha


def online_zscores(x, y,
                   beta_win=41, zscore_win=21, mtd='on',
                   mean_fn=meanvar , beta_fn=lin_reg ):
    ''' Compute sequential z-scores 
               Choice of beta calculacion: regresion / kalman filter
               Choise of averaging:
                  Using moving average window / exponential mean averaging / kalman filter
       '''
    beta = np.full_like(x, np.nan)
    spread = np.full_like(x, np.nan)
    spread_mean = np.full_like(x, np.nan)
    spread_std = np.full_like(x, np.nan)
    spread_sq = np.full_like(x, np.nan)
    zscore = np.full_like(x, np.nan)

    if mtd == 'off': # no tiene sentido aqui
        spread, beta, alpha = calculate_spread( x, y,beta_win)
        zscore, spread_mean, spread_std = off_zscore( spread, zscore_win,centred=1)
    elif mtd == 'on': # moving averages 
        for it in range(beta_win, len(x)):
            x_win = x[it-beta_win:it]
            y_win = y[it-beta_win:it]

            beta[it],alpha = beta_fn(x_win, y_win)

            spread[it] = y[it] - (beta[it] * x[it] + alpha)

            if it >= zscore_win:
                spread_win = spread[it-zscore_win+1:it+1] # incluye el it
                spread_mean[it],spread_std[it] = mean_fn(spread_win)
                zscore[it] = (spread[it] - spread_mean[it]) / spread_std[it]
    elif mtd == 'kf':
        alpha,beta = kalman_cointegration(x,y,sigma_eps=1.0, # all the alpha, beta time series
                                          sigma_eta_alpha=0.01, sigma_eta_beta=0.01)
        for it in range(beta_win, len(x)):
            x_win = x[it-beta_win:it]
            y_win = y[it-beta_win:it]

            spread[it] = y[it] - (beta[it] * x[it] + alpha[it])

            if it >= zscore_win:
                spread_win = spread[it-zscore_win+1:it+1] # incluye el it
                spread_mean[it],spread_std[it] = mean_fn(spread_win)
                zscore[it] = (spread[it] - spread_mean[it]) / spread_std[it]

    else: # exponential mean, this is purely sequential from start
        for it in range(len(x)):
            it0=max(it-beta_win,0)
            it1=max(it,beta_win)
            x_win = x[it0:it1]
            y_win = y[it0:it1]

            beta[it],alpha = beta_fn(x_win, y_win)
            spread[it] = y[it] - (beta[it] * x[it] + alpha)

            if it > 0:
                spread_mean[it], spread_std[it], spread_sq[it] = (
                    exp_mean(spread[it], spread_mean[it-1], spread_sq[it-1], zscore_win))
            else:
                spread_mean[it], spread_std[it], spread_sq[it] = exp_mean(spread[it], 0, 0, 0)
            zscore[it] = (spread[it] - spread_mean[it]) / spread_std[it]

        zscore[:beta_win]=np.nan
        
    return zscore,beta,spread,spread_mean,spread_std

def kalman_cointegration(x, y, sigma_eps=1.0, sigma_eta_alpha=0.01, sigma_eta_beta=0.01):
    ''' El estado del filtro son los parametros alpha y beta y estos son los que proyectan x en y
       y (el valor del asset) es la observacion, x (valor del asset) es parte de la H tal que
       innovacion = y - ( [1,x] * [alpha,beta] )  
    '''
    n = len(x)
    alpha_hat = np.zeros(n)
    beta_hat = np.zeros(n)
    P = np.zeros((2, 2, n))  
    
    # Initialize parameters and its covariance
    alpha_hat[0], beta_hat[0] = 0.0, y[0] / x[0] if x[0] != 0 else 0.0
    P[:, :, 0] = np.eye(2)  
    
    for t in range(1, n):
        alpha_pred = alpha_hat[t-1] # asumo modelo de persistencia
        beta_pred = beta_hat[t-1]
        P_pred = P[:, :, t-1] + np.diag([sigma_eta_alpha**2, sigma_eta_beta**2])
        
        H = np.array([1, x[t]])  # 
        S = H @ P_pred @ H.T + sigma_eps**2  
        K = P_pred @ H.T / S  # Kalman gain
        
        innovation = y[t] - (alpha_pred + beta_pred * x[t])
        alpha_hat[t] = alpha_pred + K[0] * innovation # analisis
        beta_hat[t] = beta_pred + K[1] * innovation
        P[:, :, t] = P_pred - np.outer(K, H) @ P_pred
    
    return alpha_hat, beta_hat

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
    capital =  np.zeros( nret_x.shape[0] )
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
            capital[it+1] = capital[it] * (1+0.5*(w_x * nret_x[it]-w_y *nret_y[it]))
            largo[it+1,0] = largo[it,0] * (1+w_x*nret_x[it]) 
            corto[it+1,1] = corto[it,1] * (1-w_y*nret_y[it])
        else:
            largo[it+1,0] = largo[it,0]
            corto[it+1,1] = corto[it,1]
        if ccompras[it] > 0:
            largo[it+1,1] = largo[it,1] * (1+w_y*nret_y[it]) 
            corto[it+1,0] = corto[it,0] * (1-w_x*nret_x[it])
            capital[it+1] = capital[it] * (1+ 0.5*(-w_x*nret_x[it]+w_y*nret_y[it]))
        else:
            largo[it+1,1] = largo[it,1]
            corto[it+1,0] = corto[it,0]
        if compras[it] == 0 and ccompras[it] == 0:
            capital[it+1] = capital[it]
    #capital=largo.sum(1)+corto.sum(1)
    return largo, corto,capital

def inversion(x,y,cnf,shorten=0):
    ' Hago todo el proceso on-line para un par de assets '

    x,y,nret_x,nret_y = utils.select_variables(x,y,tipo=cnf.tipo)
    mean_fn = meanvar #mean_function.get(cnf.mean_fn) # dada un string selecciono la funcion de ese nombre
    zscore0,b,s,sm,ss = online_zscores(x, y,
                                      beta_win=cnf.beta_win, zscore_win=cnf.zscore_win,
                                      mtd=cnf.mtd,
                                      mean_fn=mean_fn , beta_fn=lin_reg )
    compras0,ccompras0 = invierte( zscore0, cnf.sigma_co, cnf.sigma_ve )

    if not hasattr(cnf, 'linver_betaweight'):
        setattr(cnf, 'linver_betaweight', 0)

    beta = b if cnf.linver_betaweight else None
    largo0, corto0, capital0 = capital_invertido(nret_x,nret_y,
                                                 compras0,ccompras0,
                                                 beta=beta)

    if shorten: # problemas de memoria para simulaciones en paralelo all_pairs
        res={'capital':capital0}
    else:
        res={
            'largo':largo0, 'corto':corto0, 'capital':capital0,
            'compras':compras0, 'ccompras':ccompras0, 'zscore':zscore0,
            'beta':b, 'spread':s, 'spread_mean':sm, 'spread_std':ss }
    return res


def all_pairs(assets,company,cnf):
    ''' computations for all the pairs. Output class with all the metrics'''
    
    assets_l = list(permutations(assets, 2))
    company_l = list(permutations(company, 2))
    
    if not hasattr(cnf, 'shorten'):
        setattr(cnf, 'shorten', 1)
    return  given_pairs(assets_l,company_l,cnf,shorten=cnf.shorten)


def given_pairs(assets_l,company_l,cnf,shorten=0):
    ''' assets_l a list of tuplas of pairs of assets
        Output class with all the metrics
    '''
    res_l=[]    
    for i, (x, y) in enumerate(assets_l):
        res_d = inversion(x,y,cnf,shorten=shorten)
        res_d['company']=company_l[i]
        res_d['assets']=assets_l[i]
        
        res_l.append( res_d )
        
    return utils.Results(res_l) 


def given_pairs_multiparam(assets_l,company_l,cnf):
    ''' assets_l a list of tuplas of pairs of assets
          using different parameters for each pair
        Output class with all the metrics
    '''
    res_l=[]    
    for i, (x, y) in enumerate(assets_l):
        cnf.beta_win =cnf.params_l[i][0]
        cnf.zscore_win =cnf.params_l[i][1]
        cnf.sigma_co =cnf.params_l[i][2]
        cnf.sigma_ve =cnf.params_l[i][3]
        res_d = inversion(x,y,cnf)
        res_d['company']=company_l[i]
        res_d['assets']=assets_l[i]
        
        res_l.append( res_d )
        
    return utils.Results(res_l) 


