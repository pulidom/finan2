import numpy as np
import pickle
from numpy.lib.stride_tricks import sliding_window_view
def rms(E):
    'E[nt,nx] is already an error/anomaly ' 
    return (np.sqrt((E*E).mean(1))).mean()

#------------------------------------------------------
def spread(X):
  " X[nt,[nx,npa]] "
  n=X.shape
  xm=X.mean(-1,keepdims=True)
  return np.sqrt((np.square(X-xm)).mean(-1))

#------------------------------------------------------
def obj2dict(obj):
    " Changes from object variables to dictionary"
    a=dict((key, value) for key, value in obj.__dict__.items() 
         if not callable(value) and not key.startswith('__'))
    return a

#------------------------------------------------------
def mrdiv(b,A):
  """ Matrix right division  K= P Q^-1  --> K=mrdiv(P,Q)
  
  """
  return np.linalg.solve(A.T,b.T).T

#--------------------------------------------------
def meanper(X):
    " Compute mean and perturbation X[nx,nem]"
    xm = X.mean(-1, keepdims=True)
    return xm,X-xm

#--------------------------------------------------
def cova(X,Y):
  "Determine the sample cov of two fields dim(state,ens members)"

  nem=X.shape[1]

  _,dX=meanper(X)
  _,dY=meanper(Y)
  covXY=(dX @ dY.T)/(nem-1.)

  return covXY

#--------------------------------------------------
def PaWa_svd(B):
    "Compute inv and inv sqrt using SVD. Used for ETKF"

    U, s, V = np.linalg.svd(B)#, full_matrices=True)
#    U, s, V = sla.svd(B)#, full_matrices=True)

    n=V.shape[-2]
    m=V.shape[-1]

    invs = 1/s[:n]        
    invsqs = s[:n]**(-0.5)

    Pa=V.T @ (invs * U).T

    Wa = (m-1.)**.5 *(V.T @ (invsqs * U).T)

    return Pa, Wa

#--------------------------------------------------
def PaWa_eig(B):
    "Compute inv and inv sqrt using eigenvectors. Used for ETKF"

    s, V = np.linalg.eigh(B)

    n=V.shape[-2]
    m=V.shape[-1]

    invs = 1/s[:n]        
    invsqs = s[:n]**(-0.5)

    Pa=V @ np.diag(invs) @ V.T
    Wa = (m-1.)**.5 *(V @ np.diag(invsqs) @ V.T)

    return Pa, Wa

#--------------------------------------------------
def saveout(fname,res):
    ' Save data of the analysis or any other dictionary'
    fileObj = open(fname+'.pkl','wb') 
    pickle.dump(res,fileObj, pickle.HIGHEST_PROTOCOL)
    fileObj.close()

#--------------------------------------------------    
def loadout(fname):
    ' Load data of the analysis or any other dictionary'
    fileObj = open(fname+'.pkl','rb') 
    res = pickle.load(fileObj)
    fileObj.close()
    return res

#--------------------------------------------------    
def sqrt_svd(A):  
    "Returns the square root matrix by SVD"

    U, s, V = np.linalg.svd(A)#, full_matrices=True)
    
    sqrts = np.sqrt(s)
    n = np.size(s)
    sqrtS = np.zeros((n,n))
    sqrtS[:n, :n] = np.diag(sqrts)
    
    sqrtA=np.dot(V.T,np.dot(sqrtS, U.T))

    return sqrtA

#--------------------------------------------------    
def sqrt_eig(A):
    "Compute square root using eigenvectors"

    s, V = np.linalg.eigh(A)#assume simmetry/ pos-sem-def  of A

    s = np.maximum(s,0)
    sqrts = np.sqrt(s)
    
    sqrtA=(V * sqrts) @ V.T

    return sqrtA

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
