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
import optimal_transport as ot
from scipy.stats import rankdata
import copula

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
                   mean_fn=meanvar , beta_fn=lin_reg, eps = 1.e-10 ):
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
                zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it]+eps)
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
    elif mtd=='ot':
        for it in range(beta_win, len(x)):
            # Ventana de entrenamiento OT
            x_win = x[it - beta_win:it+1]
            y_win = y[it - beta_win:it+1]

            y_sample, *_ = ot.ot_barycenter_solver(y_win, x_win, n_iter=100)
            spread[it]=y_sample[-1]
            spread_mean[it], spread_std[it] = np.mean(y_sample), np.std(y_sample)
            zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it] + eps)
#            if it >= zscore_win:
#                spread_win = spread[it-zscore_win+1:it+1] # incluye el it
#                spread_mean[it],spread_std[it] = mean_fn(spread_win)
#                zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it]+eps)

## Calcula el rango / (N + 1) para evitar p=0 y p=1
#N = len(x_sample_no_normal)
#percentiles = rankdata(x_sample_no_normal) / (N + 1)

## 2. Aplicar la función de Percent Point (Inversa de la CDF Normal)
#z_score_cdf = norm.ppf(percentiles)

    elif mtd=='ot2':
        for it in range(beta_win, len(x)):
            # Ventana de entrenamiento OT
            x_win = x[it - beta_win:it+1]
            y_win = y[it - beta_win:it+1]

            y_sample,  s, U, Vt, Qz, Bz, By, centers_z, lambda_val = ot.ot_barycenter_solver(y_win, x_win, n_iter=100)
            x_sample = ot.simulate_conditional(y_sample, y_win[-1], Qz, Bz, By, centers_z, lambda_val, s, U, Vt)
            spread_mean[it], spread_std[it] = np.mean(x_sample), np.std(x_sample)
#            zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it] + eps)
            zscore[it] = (x_win[-1] - spread_mean[it]) / (spread_std[it] + eps)
#            if it >= zscore_win:
#                spread_win = spread[it-zscore_win+1:it+1] # incluye el it
#                spread_mean[it],spread_std[it] = mean_fn(spread_win)
#                zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it]+eps)
    elif mtd=='copula':
        for it in range(beta_win, len(x)):
            # Ventana de entrenamiento OT
            x_win = x[it - beta_win:it+1]
            y_win = y[it - beta_win:it+1]

            spread0 = copula.copula_zscore(x_win,y_win)
            spread[it]=spread0[-1]
            zscore[it] =spread0[-1]
#            if it >= zscore_win:
#                spread_win = spread[it-zscore_win+1:it+1] # incluye el it
#                spread_mean[it],spread_std[it] = mean_fn(spread_win)
#                zscore[it] = (spread[it] - spread_mean[it]) / spread_std[it]
                #spread_mean[it], spread_std[it] = (np.mean(spread0[-zscore_win:]),
                #                                   np.std(spread0[-zscore_win:]))
                #zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it] + eps)
                #
##            print('z',zscore[it]) 

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
            zscore[it] = (spread[it] - spread_mean[it]) / (spread_std[it] + eps )

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
    largo0, corto0, capital0,retorno0 = capital_invertido(nret_x,nret_y,
                                                 compras0,ccompras0,
                                                 beta=beta)

    if shorten: # problemas de memoria para simulaciones en paralelo all_pairs
        res={'capital':capital0}
    else:
        print('')
        res={
            'largo':largo0, 'corto':corto0, 'capital':capital0, 'retorno':retorno0,
            'compras':compras0, 'ccompras':ccompras0, 'zscore':zscore0,
            'beta':b, 'spread':s, 'spread_mean':sm, 'spread_std':ss }
    return res

def inversion_zscore(zscore0,nret_x,nret_y,cnf,shorten=0):
    
    compras0,ccompras0 = invierte( zscore0, cnf.sigma_co, cnf.sigma_ve )
    largo0, corto0, capital0,retorno0 = capital_invertido(nret_x,nret_y,
                                                          compras0,ccompras0,
                                                          beta=None)

    if shorten: # problemas de memoria para simulaciones en paralelo all_pairs
        res={'capital':capital0}
    else:
        print('')
        res={
            'largo':largo0, 'corto':corto0, 'capital':capital0, 'retorno':retorno0,
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
        #res_d['assets']=assets_l[i]
        res_d['asset_x']=x
        res_d['asset_y']=y
        
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

### con weights pero no terminado
def invierte_weights(zscore,sigma_co=1.5,sigma_ve=0.5):
    ''' Determina los intervalos temporales de compra venta en una serie
        single time series
       '''

    weights=np.zeros(zscore.shape[0], dtype=np.int8)
    
    band,cband=0,0
    for it in range(zscore.shape[0]):
        if band: # poseo el activo
            if zscore[it] > sigma_ve:
                weights[it]=1 # mantengo
            else:
                band=0 # vendo
        else: # no poseo el activo
            if zscore[it] > sigma_co:
                band=1 # compro
                weights[it]=1
        # posiciones en corto
        if cband:
            if zscore[it] < - sigma_ve:
                weights[it]=-1 # mantengo
            else:
                cband=0 # vendo
        else:
            if zscore[it] < - sigma_co:
                cband=1 # compro
                weights[it]=-1
    return weights

def capital_invertido_weights(nret_x,nret_y,weights,beta=None):
    ''' invierto el capital con pares
        divide la inversion en forma equitativa o con beta weights
        Manejo con weights la entrada y salida de posiciones
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
        
        if weights[it] > 0:
            retorno[it+1] = 0.5*(w_x * nret_x[it]-w_y *nret_y[it])
            capital[it+1] = capital[it] * (1+retorno[it+1])
            largo[it+1,0] = largo[it,0] * (1+w_x*nret_x[it]) 
            corto[it+1,1] = corto[it,1] * (1-w_y*nret_y[it])
        elif weights[it] < 0:
            retorno[it+1] = 0.5*(-w_x*nret_x[it]+w_y*nret_y[it])
            largo[it+1,1] = largo[it,1] * (1+w_y*nret_y[it]) 
            corto[it+1,0] = corto[it,0] * (1-w_x*nret_x[it])
            capital[it+1] = capital[it] * (1+retorno[it+1])
        else:
            largo[it+1,1] = largo[it,1]
            corto[it+1,0] = corto[it,0]
            capital[it+1] = capital[it]
            
    return largo, corto,capital,retorno

def inversion_weights(x,y,cnf,shorten=0):
    ' Hago todo el proceso on-line para un par de assets '

    x,y,nret_x,nret_y = utils.select_variables(x,y,tipo=cnf.tipo)
    mean_fn = meanvar #mean_function.get(cnf.mean_fn) # dada un string selecciono la funcion de ese nombre
    zscore0,b,s,sm,ss = online_zscores(x, y,
                                      beta_win=cnf.beta_win, zscore_win=cnf.zscore_win,
                                      mtd=cnf.mtd,
                                      mean_fn=mean_fn , beta_fn=lin_reg )
    weights0 = invierte( zscore0, cnf.sigma_co, cnf.sigma_ve )

    if not hasattr(cnf, 'linver_betaweight'):
        setattr(cnf, 'linver_betaweight', 0)

    beta = b if cnf.linver_betaweight else None
    largo0, corto0, capital0,retorno0 = capital_invertido(nret_x,nret_y,
                                                 weights0,
                                                 beta=beta)

    if shorten: # problemas de memoria para simulaciones en paralelo all_pairs
        res={'capital':capital0}
    else:
        res={
            'largo':largo0, 'corto':corto0, 'capital':capital0, 'retorno':retorno0,
            'weights':weights0, 'zscore':zscore0,
            'beta':b, 'spread':s, 'spread_mean':sm, 'spread_std':ss }
    return res


### DE ACÁ PARA ABAJO ES INVENTO DE GASTON

def volume_weight(res_,cnf,volume,assets,company):
    
    w_volumen = np.zeros(res_.retorno.shape)
    #print('w_volumen',w_volumen)

    #assets_l = list(permutations(assets, 2))
    #company_l = list(permutations(company, 2))
    #print(res_.company)
    #print(len(res_.company[:,0]))
    for par in range(len(res_.company[:,0])): 
        c0 , c1 = res_.company[par,0], res_.company[par,1]
        ubic_0 , ubic_1 = np.where(company == c0)[0][0] , np.where(company == c1)[0][0]
        p0     , p1     = res_.assets[par,0,:] , res_.assets[par,1,:]
        v0     , v1     = volume[ubic_0, :]*p0 , volume[ubic_1, :]*p1
        volumen_teorico = min(np.mean(v0[:-cnf.Njump]),np.mean(v1[:-cnf.Njump])) 
        
        for dia in range(1, res_.retorno.shape[1]):  # arranca en 1 por el dia-1
            # volumen * precio
            # cap de 10M usd
            v0i,v1i = v0[dia-1],  v1[dia-1]
            volumen_actual = min(v0i,v1i)
            
            if volumen_actual>10_000_000:
                ratio = volumen_teorico / volumen_actual
                if ratio>2:
                    ratio=2
                #w_volumen[par, dia] = res_.retorno[par, dia-1] * ratio
                w_volumen[par, dia] = ratio
            else:
                #w_volumen[par, dia] = 1e-18#da problemas si es =0
                w_volumen[par, dia] = 0
    return w_volumen



def volatility_weight(res_,cnf,volume,assets,company):
    
    w_volatilidad = np.zeros(res_.retorno.shape)
    
    for par in range(len(res_.company[:,0])):
        c0 , c1 = res_.company[par,0], res_.company[par,1]
        ubic_0 , ubic_1 = np.where(company == c0)[0][0] , np.where(company == c1)[0][0]
        p0     , p1     = res_.assets[par,0,:] , res_.assets[par,1,:]
        v0     , v1     = volume[ubic_0, :]*p0 , volume[ubic_1, :]*p1
        
        volatilidad_teorica = np.abs(res_.spread_mean[par,-cnf.Njump])
        
        for dia in range(1, res_.retorno.shape[1]):  # arranca en 1 por el dia-1

            volatilidad_actual  = res_.spread[par,dia]
            zscore = res_.zscore[par,dia]
            p0i,p1i= p0[dia-1], p1[dia-1]
            v0i,v1i = v0[dia-1],  v1[dia-1]
            volumen_actual = min(v0i,v1i)
            
            price = min(p0i,p1i)
            
            if volumen_actual>10_000_000:
                ratio=np.abs(zscore*volatilidad_teorica/(price*volatilidad_actual))
                if ratio>1:
                    ratio=1
                w_volatilidad[par, dia] = ratio
            else:
                w_volatilidad[par, dia] = 0
    
    return w_volatilidad

def given_pairs_weighted(assets_l, company_l, cnf, res_, volume=None,weight_met=volume_weight):
    """
    Corre la estrategia en todos los pares seleccionados y calcula métricas agregadas ponderadas
    por volumen relativo (estimado internamente), manteniendo el peso constante durante la posición.
    
    assets: matriz de precios
    company: lista de nombres de empresas
    cnf: configuración
    res_: resultado de una corrida previa de all_pairs (para identificar pares y retorno)
    volume: matriz de volumen (empresa x tiempo)

    Retorna: objeto Results con capital_ponderado y retorno_ponderado
    """
    #assets_l = list(permutations(assets_l, 2))
    weights = weight_met(res_,cnf,volume,assets_l,company_l)
    #print('www',weights.shape)
    company_l = list(permutations(company_l, 2))
    #for i, (x, y) in enumerate(assets_l):
    #    print(i)
    
    assets_l = res_.assets

    n_pairs = len('assets_l')
    n_days = weights.shape[1]
    res_l = []
    #print('weit',weights.shape)
    
    pesos_congelados = np.zeros_like(weights)
    
    for i, (x, y) in enumerate(assets_l):
        res_d = inversion(x, y, cnf, shorten=0)
        res_d['company'] = company_l[i]
        res_d['assets'] = assets_l[i]
        res_l.append(res_d)

        # Congelar pesos durante posición abierta
        peso_actual = 0
        pos = False
        for t in range(n_days):
            compra = res_d['compras'][t]
            ccompra = res_d['ccompras'][t]
            
            if compra or ccompra:
                if not pos:
                    peso_actual = weights[i, t]
                    pos=True
            else:
                pos=False
                peso_actual = peso_actual
            pesos_congelados[i, t] = peso_actual

    # Normalizar pesos por día
    #print(pesos_congelados[2])
    suma_pesos = pesos_congelados.sum(axis=0) + 1e-18
    #print(suma_pesos.shape)
    pesos_normalizados = pesos_congelados / suma_pesos[:]
    #print(np.max(pesos_normalizados))
    #print(np.min(pesos_normalizados))
    # Capital y retorno ponderados
    capital_matrix = np.array([res['capital'] for res in res_l])
    retorno_matrix = np.array([res['retorno'] for res in res_l])
    #capital_ponderado = (capital_matrix * pesos_normalizados).sum(axis=0)
    capital_ponderado = capital_matrix
    retorno_ponderado = (retorno_matrix * pesos_normalizados).sum(axis=0)

    # Empaquetar resultados
    resultado_final = utils.Results(res_l)
    resultado_final.capital_ponderado = capital_ponderado
    resultado_final.retorno_ponderado = retorno_ponderado
    resultado_final.pesos_usados = pesos_congelados
    resultado_final.pesos_normalizados = pesos_normalizados


    return resultado_final

def ordenar_pares(res,volume,metrica,cnf,inverso_flag=False,capear_por_volumen=False):

    if capear_por_volumen:
        for par in range(len(res.company[:,0])): 

            c0 , c1 = res.company[par,0], res.company[par,1]
            #print(np.where(company == c0)[0][0])
            ubic_0 , ubic_1 = np.where(res.company == c0)[0][0] , np.where(res.company == c1)[0][0]
            p0     , p1     = res.assets[par,0,:] , res.assets[par,1,:]
            v0     , v1     = volume[ubic_0, :] * p0 , volume[ubic_1, :] * p1
            volumen_teorico = min(np.mean(v0[:-cnf.Njump]),np.mean(v1[:-cnf.Njump])) 
            
            if volumen_teorico > 10_000_000:
                if inverso_flag:
                    metrica[par]+=-9999
                else:
                    metrica[par]+=9999

    idx = np.argsort(metrica)[:cnf.nsel]
    res.reorder(idx)

    return 
