''' Codigos computacionales. medias etc
'''
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
def select_variables(x,y,tipo='asset'):
    ''' Selecciona la variable y el retorno normalizado correspondiente
        para calculo de ganancia
         En el caso de usar retornos como x e y selecciono para retornos normalizados
       los del dia siguiente.
    '''
    
    if tipo=='asset':
        nret_x = (x[1:]-x[:-1])/x[:-1]
        nret_y = (y[1:]-y[:-1])/y[:-1]
    elif tipo=='log':
        nret_x = (x[1:]-x[:-1])/x[:-1]
        nret_y = (y[1:]-y[:-1])/y[:-1]
        x=np.log(x)
        y=np.log(y)
    elif tipo=='return':
        nret_x = (x[2:]-x[1:-1])/x[1:-1]
        nret_y = (y[2:]-y[1:-1])/y[1:-1]
        x=x[1:]-x[:-1]
        y=y[1:]-y[:-1]
    elif tipo=='log_return':
        nret_x = (x[2:]-x[1:-1])/x[1:-1] # tengo que ver en el dia siguiente
        nret_y = (y[2:]-y[1:-1])/y[1:-1]
        x=np.log(x[1:])-np.log(x[:-1])
        y=np.log(y[1:])-np.log(y[:-1])
    elif tipo=='ratio':
        x1= x/y
        y = x1
        x = x1
        nret_x = (x[1:]-x[:-1])/x[:-1]
        nret_y = (y[1:]-y[:-1])/y[:-1]
        
    return x,y,nret_x,nret_y

class Results:
    ''' Transform a list of dictionaries in a class of np.arrays '''
    def __init__(self, list_of_dicts):
        keys = list_of_dicts[0].keys()
        for key in keys:
            setattr(self, key, np.array([d[key] for d in list_of_dicts]) )

    def reorder(self, idx):
        ''' reorder the arrays following the idx indices (capital, corto, etc) '''
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(self, key, value[idx])
                
#--------------------------------------------------    
def obj2dict(obj):
    " Changes from object variables to dictionary"
    a=dict((key, value) for key, value in obj.__dict__.items() 
         if not callable(value) and not key.startswith('__'))
    return a
                
class dict2obj:
    " Dictionary 2 objetc use: a=dict2obj(**dictio) "
    def __init__(self, **entries):
        self.__dict__.update(entries)

def lin_reg(x,y):
    slope, intercept, _, _, _ = linregress(x, y)
    return slope, intercept
def lin_reg_alpha0(x,y):
    slope, intercept, _, _, _ = linregress(x, y)
    return slope, 0


def meanvar(spread):
    return np.nanmean(spread),np.nanstd(spread)

mean_function={'meanvar':meanvar} # define funcion por default (usando un dictio).


def exp_mean(x, ema0, ema_sq0, period):
    """
    Computes the exponential moving average (EMA) of a 1D array.
       """
    alpha = 2 / (period + 1)

    if ema0==0 and ema_sq0==0: # initialization
        ema=x
        ema_sq=x**2
        std=0
    else:
        ema = alpha * x + (1 - alpha) * ema0
        ema_sq = alpha * x**2 + (1 - alpha) * ema_sq0
        variance = ema_sq - ema**2
        std = np.sqrt(max(variance, 0))  # Avoid negative variance due to float error

    return ema, std, ema_sq

#--------------------------------------------------    
def erolling(data, period):
    """
    Computes the exponential moving average (EMA) of a 1D array.
        """
    alpha = 2 / (period + 1)

    ema, ema_sq, std= np.zeros((3,*data.shape))

    ema[0] = data[0]
    ema_sq[0] = data[0]**2
    std[0] = 0.0
    
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
        ema_sq[t] = alpha * (data[t]**2) + (1 - alpha) * ema_sq[t-1]
        variance = ema_sq[t] - ema[t]**2
        std[t] = np.sqrt(max(variance, 0))  # Avoid negative variance due to float error

    return ema, std

#--------------------------------------------------    
def rolling(arr, window, func, padding=True):
    """
    Mimics pandas' rolling().apply() using NumPy.

    Parameters:
    - arr: 1D NumPy array
    - window: Integer, size of the rolling window
    - func: Function to apply to each rolling window (e.g., np.mean, np.std)
    - padding: If True, returns array of same size with np.nan padding

    Returns:
    - result: NumPy array of rolled values
    """
    arr = np.asarray(arr)
    if window > len(arr):
        raise ValueError("Window size must be less than or equal to array length.")
    
    result = np.array([func(arr[i:i+window]) for i in range(len(arr) - window + 1)])
    
    if padding:
        # Pad the beginning with NaNs to match original length
        pad = np.full(window - 1, np.nan)
        result = np.concatenate([pad, result])
    return result

#--------------------------------------------------    
def crolling(arr, window, func):
    ''' central rolling window '''
    
    if window % 2 == 0:
        raise ValueError("Window size must be odd for perfect centering.")
    
    half_window = window // 2
    # Create full padding on both sides
    padded = np.pad(arr, pad_width=half_window, mode='edge')  # or mode='reflect'/'constant'

    # Create rolling windows
    windows = sliding_window_view(padded, window_shape=window)

    # Apply the function across axis=1
    result = np.apply_along_axis(func, axis=1, arr=windows)
    return result

#--------------------------------------------------    
def rolling_meanvar(spread,window,centred=0):
    if centred==1:
        spread_mean = crolling(spread,window,np.mean)    
        spread_std = crolling(spread,window,np.std)
    elif centred==-1:
        spread_mean,spread_std = erolling(spread,window)
    else: #==0
        spread_mean = rolling(spread,window,np.mean)    
        spread_std = rolling(spread,window,np.std)
    return spread_mean,spread_std

#--------------------------------------------------    
def calc_startend(bool_arr):
    ''' Dado un array booleano determina 
        Fecha inicio y fin de una posicion/es'''

    start_indices,end_indices=[],[]    
    for ivar in range(bool_arr.shape[1]):
        diff = np.diff(bool_arr[:,ivar].astype(int))
        start_indice = np.where(diff == 1)[0]
        end_indice = np.where(diff == -1)[0]

        # Handle edge cases where the boolean array starts or ends with True
        if bool_arr[0,ivar]:
            start_indice = np.insert(start_indice, 0, 0)
        if bool_arr[-1,ivar]:
            end_indice = np.append(end_indice, bool_arr.shape[0]-1)
        start_indices.append(start_indice)
        end_indices.append(end_indice)
        
    return start_indices, end_indices

def returns_from(capital,jday):
    ''' Recalculo el rendimiento a partir de un dia '''
    rets=(capital.T[jday+1:]-capital.T[jday:-1])/capital.T[jday:-1]
    caps=np.zeros((rets.shape[0]+1,rets.shape[1]))

    caps[0,:]=100
    for i in range(1,caps.shape[0]):
        caps[i] = caps[i-1]*(1+rets[i-1])
    return caps.T
