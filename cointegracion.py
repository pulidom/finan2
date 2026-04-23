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
    assets= [x,y]
    res={  'assets':assets,'asset_x': x,'asset_y': y,'largo':largo0, 'corto':corto0,
           'capital':capital0, 'retorno':retorno0,
           'compras':compras0, 'ccompras':ccompras0, 'zscore':zscore0,
           'beta':b, 'spread':s, 'spread_mean':sm, 'spread_std':ss }
    return res

import pandas as pd

def zscore_fixed_beta(x, y, alpha, beta, zscore_win=21, eps=1.e-10):
    """
    Calcula el Z-Score asumiendo que alpha y beta ya están fijos desde In-Sample (No Look-ahead bias).
    Las medias y varianzas rodantes utilizan estrictamente datos hasta t-1 (Shift 1).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    spread = y - (beta * x + alpha)
    
    # Usar pandas para veloz cálculo en C y prevenir loop lento
    spread_s = pd.Series(spread)
    
    # .shift(1) garantiza evitar total look-ahead hacia el día de hoy
    spread_mean = spread_s.rolling(window=zscore_win, min_periods=1).mean().shift(1).values
    spread_std = spread_s.rolling(window=zscore_win, min_periods=1).std().shift(1).values
    
    # Evitamos NaNs en el primer index del shift
    spread_mean[0] = spread[0]
    spread_std[np.isnan(spread_std)] = eps # Prevenir warnings por pocos datos init
    spread_std[spread_std == 0] = eps
    
    zscore = (spread - spread_mean) / spread_std
    
    # Devolvemos un par de arreglos (beta mock array para consistencia si se llegara a pedir)
    b_arr = np.full_like(zscore, beta)
    
    return zscore, b_arr, spread, spread_mean, spread_std

def invierte_con_estadisticas_fijas(
    spread,
    spread_mean,
    spread_std,
    sigma_co=1.5,
    sigma_ve=0.5,
    sigma_stop=np.inf,
    eps=1.e-10,
):
    """
    Usa estadísticos rodantes para entrar, pero una vez abierta la posición
    congela mu y sigma hasta que el z-score de salida indique cierre o se
    alcance un umbral de seguridad por ruptura persistente.
    """
    spread = np.asarray(spread)
    spread_mean = np.asarray(spread_mean)
    spread_std = np.asarray(spread_std)
    sigma_stop = float(sigma_stop)
    if np.isfinite(sigma_stop):
        sigma_stop = max(abs(sigma_stop), abs(float(sigma_co)) + eps, abs(float(sigma_ve)) + eps)
    else:
        sigma_stop = np.inf

    compras = np.zeros(spread.shape[0], dtype=bool)
    ccompras = np.zeros(spread.shape[0], dtype=bool)
    zscore_eval = np.full(spread.shape[0], np.nan)

    band, cband = 0, 0
    long_mean = long_std = None
    short_mean = short_std = None

    for it in range(spread.shape[0]):
        mean_t = spread_mean[it] if np.isfinite(spread_mean[it]) else spread[it]
        std_t = spread_std[it] if np.isfinite(spread_std[it]) and spread_std[it] > eps else eps
        base_z = (spread[it] - mean_t) / std_t
        zscore_eval[it] = base_z

        if band:
            z_long = (spread[it] - long_mean) / (long_std + eps)
            zscore_eval[it] = z_long
            if np.isfinite(sigma_stop) and z_long >= sigma_stop:
                band = 0
                long_mean = None
                long_std = None
            elif z_long > sigma_ve:
                compras[it] = True
            else:
                band = 0
                long_mean = None
                long_std = None
        else:
            if base_z > sigma_co:
                band = 1
                long_mean = mean_t
                long_std = std_t
                compras[it] = True

        if cband:
            z_short = (spread[it] - short_mean) / (short_std + eps)
            zscore_eval[it] = z_short
            if np.isfinite(sigma_stop) and z_short <= -sigma_stop:
                cband = 0
                short_mean = None
                short_std = None
            elif z_short < -sigma_ve:
                ccompras[it] = True
            else:
                cband = 0
                short_mean = None
                short_std = None
        else:
            if base_z < -sigma_co:
                cband = 1
                short_mean = mean_t
                short_std = std_t
                ccompras[it] = True

    return compras, ccompras, zscore_eval

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

def capital_invertido(nret_x, nret_y, compras, ccompras, beta=None, cost=0.005):
    ''' 
    Versión VECTORIZADA de Inversión de capital para Pair Trading.
    Aceleración sustancial sobre el bucle For original. 
    '''
    n = nret_x.shape[0]
    m = n - 1 # El bucle evalúa compras[it] y calcula retorno[it+1], iterando n-1 veces.
    
    if m <= 0:
        return np.zeros((1, 2)), np.zeros((1, 2)), np.array([100.0]), np.zeros(1)
        
    nret_x_c = nret_x[:m]
    nret_y_c = nret_y[:m]
    
    if beta is None:
        w_x = np.ones(m)
        w_y = np.ones(m)
    else:
        beta = np.asarray(beta[:m])
        if beta.ndim == 0:
            beta = np.full(m, beta)
        w_x = np.abs(beta) / (1 + np.abs(beta))
        w_y = 1 / (1 + np.abs(beta))
        
    c_int = compras[:m].astype(int)
    cc_int = ccompras[:m].astype(int)
    
    c_shift = np.concatenate(([0], c_int[:-1]))
    cc_shift = np.concatenate(([0], cc_int[:-1]))
    
    tc_compras = cost * ((c_int > 0) & (c_shift == 0)).astype(float) + \
                 cost * ((c_int == 0) & (c_shift > 0)).astype(float)
                 
    tc_ccompras = cost * ((cc_int > 0) & (cc_shift == 0)).astype(float) + \
                  cost * ((cc_int == 0) & (cc_shift > 0)).astype(float)
                  
    ret_long = 0.5 * (w_x * nret_x_c - w_y * nret_y_c)
    ret_short = 0.5 * (-w_x * nret_x_c + w_y * nret_y_c)
    
    retorno = np.zeros(n)
    
    is_long = c_int > 0
    is_short = cc_int > 0
    
    active_ret_per_day = np.zeros(m)
    active_ret_per_day[is_long] = ret_long[is_long]
    active_ret_per_day[is_short] = ret_short[is_short]
    
    # Deduct transaction costs
    active_ret_per_day = active_ret_per_day - tc_compras - tc_ccompras
    
    retorno[1:] = active_ret_per_day
    capital = 100.0 * np.cumprod(1 + retorno)
    
    # Placeholder para retrocompatibilidad
    largo, corto = np.zeros((n, 2)), np.zeros((n, 2))
            
    return largo, corto, capital, retorno
