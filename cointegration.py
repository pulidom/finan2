'''  Calculate cointegration tests and metrics  

'''
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc as hurste
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.linear_model import LinearRegression
import utils
from utils import (rolling, erolling, crolling,
                   rolling_meanvar, exp_mean, mean_function, meanvar,
                   lin_reg,lin_reg_alpha0)

def calculate_spread_off( x, y):
    """
    Calcula el spread entre dos activos usando regresion lineal

    Ojo porque calcula alpha y beta en la misma ventana de tiempos que la prediccion
    """
    x=x[:,None]
    model = LinearRegression()
    model.fit(x, y)
    beta = model.coef_  # Similar to results.params[1:] in statsmodels
    alpha = model.intercept_  # Similar to results.params[0] in statsmodels
    spread = y - model.predict(x)

    return spread, (alpha, beta)

def off_zscore( spread, window, centred=0 ):
    spread_mean, spread_std = rolling_meanvar(spread,window,centred)
    zscore = (spread - spread_mean) / spread_std
    return zscore, spread_mean, spread_std

def adf_test(y):
    result = adfuller(y)
    adf_statistic = result[0]
    p_value = result[1]
    return result[1]#,result[0]

def hurst_rs(ts, max_lag=100, min_lag=20):
    ' Calculo tradicional (usar libreria hurst)'
    ts = np.asarray(ts)
    N = len(ts)
    lags = np.arange(min_lag, max_lag)

    rs_vals = []

    for lag in lags:
        n_segments = N // lag
        if n_segments == 0:
            continue

        data = ts[:n_segments * lag].reshape(n_segments, lag)
        mean = data.mean(axis=1, keepdims=True)
        dev = data - mean
        cum_dev = np.cumsum(dev, axis=1)
        R = np.ptp(cum_dev, axis=1)  # range = max - min
        S = data.std(axis=1, ddof=1)

        valid = S > 0
        R_S = R[valid] / S[valid]
        rs_vals.append(R_S.mean())

    log_lags = np.log(lags[:len(rs_vals)])
    log_rs = np.log(rs_vals)
    H, intercept = np.polyfit(log_lags, log_rs, 1)

    return H

def estimate_hurst(log_series, max_lag=100):
    ' Basado en la propuesta del libro de Chan'
    lags = range(1, max_lag + 1)
    D_tau = [np.mean(np.diff(log_series, lag)**2) for lag in lags]
    log_lags = np.log(lags)
    log_D = np.log(D_tau)
    slope, _, _, _, _ = linregress(log_lags, log_D)
    return slope / 2

def half_life(spread):
    """
    Estimate the half-life of mean reversion using linear regression.

    Parameters:
        spread (array-like): Spread time series (e.g., log price ratio)

    Returns:
        half_life (float): Estimated half-life in time steps
    """
    spread = np.asarray(spread)
    z_lag = spread[:-1]
    delta_z = np.diff(spread)

    slope, intercept, _, _, _ = linregress(z_lag, delta_z)

    rho = 1 + slope
    if abs(rho) < 1:
        half_life = np.log(0.5) / np.log(abs(rho))
    else:
        half_life = np.inf  # Not mean-reverting
    return half_life

def half_life_penalty(hl):
    """
    Piecewise penalty for half-life:
      - 0 penalty if 7 <= hl <= 15
      - linear from 1 to 0 as hl goes from 3 to 7 (penalty decreases)
      - linear from 0 to 1 as hl goes from 15 to 21 (penalty increases)
      - penalty = 1 if hl < 3 or hl > 21
    Quiero castigar los periodos si no estan entre 7-15 dias de periodo
    """
    if hl < 3:
        return 1.0
    elif 3 <= hl < 7:
        # penalty decreases from 1 to 0
        return (7 - hl) / 4
    elif 7 <= hl <= 15:
        return 0.0
    elif 15 < hl <= 21:
        # penalty increases from 0 to 1
        return (hl - 15) / 6
    else:
        return 1.0

def stats(assets_l,tipo):
    ''' compute statistics with the spread in a period'''
    
#    assets_l = list(permutations(assets, 2))
#    company_l = list(permutations(company, 2))
    pvalue0=[]
    hurst0=[]
    half_life0=[]
    score0=[]
    johansen0=[]
    for (x,y) in assets_l:
        x,y,_,_ = utils.select_variables(x,y,tipo)
        spread ,_= calculate_spread_off(x,y)
        p=adf_test(spread)
        pvalue0.append(p)
        H,_,_ = hurste(spread)
        hurst0.append(H)
        hl= half_life(spread)
        half_life0.append(hl)
        score0.append(cointegration_score(p,H,hl))
        result = coint_johansen(np.array([x,y]).T, det_order=0, k_ar_diff=1)
        johansen0.append(result.lr1[0]-result.cvt[0, 1])
        #positive values reject H0 non-stationarity
        #print("Trace test critical values (90%, 95%, 99%):", result.cvt)
    class metrics:
        pvalue=pvalue0;johansen=johansen0;hurst=hurst0;half_life=half_life0;score=score0
    return metrics

def sharpe_ratio(capital):
    '''  annual sharpe ratio '''
    num_days = len(capital)
    num_years = num_days / 252  #  252 días de trading por año        
    total_return = (capital[-1] /  capital[0]) - 1
    cagr = (1 + total_return) ** (1 / num_years) - 1
    risk_free_rate = 0.02  # Tasa libre de riesgo anualizada
    daily_returns = (capital[1:]- capital[:-1])/capital[:-1] 
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility
    return sharpe 

def cointegration_score(adf_pval, hurst, half_life):
    ''' Metrica ad-hoc que combina p-value, hurst, y half life '''
    pval_score = np.clip(adf_pval, 0, 1)
    hurst_score = np.clip(hurst, 0, 1)
    hl_score = half_life_penalty(half_life)
    
    score = 0.4 * 10 * pval_score + 0.3 * hurst_score + 0.3 * hl_score
    return score
