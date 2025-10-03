""" Generate data from a dynamical system or saved data suitable for pytorch 

"""
import sys;  sys.path.insert(0, '../')
import numpy as np
import numpy.random as rnd
from numpy.lib.stride_tricks import sliding_window_view
import torch as tor
from torch.utils.data import Dataset
from torch.utils import data 

class DriveData(Dataset):
    """ Dada la serie de tiempos la deja lista para utilizar con el DataLoader de pytorch 
    nt_in=1,nt_out=1,  number of time of input and of output variables
     los chunks se eligen cada nt_in tiempos
    """
    def __init__(self,dat, # time series
                 device='cpu',
                 nt_in=1,nt_out=1, # number of time of input and of output
                 jvar_in=[0],jvar_out=[0], # input and output variables
                 jcovariates=[1,2], # covariate variables
                 normalize=None,
                 ):

        
        if normalize is not None:
            if normalize == 'gauss':                
                dat=self.normalize_gauss(dat)
            elif normalize == 'min-max':
                dat=self.normalize_minmax(dat)

        #, self.covariates        
        self.xs,self.ys = self.chunks(dat,nt_in,nt_out,
                                      jvar_in,jvar_out,jcovariates)
        # se ponen ambas variables y covariates en el xs
        self.xs = self.normalize_minmax_window(self.xs)
        self.ys = self.normalize_minmax_window(self.ys)
        self.v_scl= self.xs.mean(1)
        self.xs = self.normalize_minmax_window(self.xs)/self.v_scl[:,None,:]
        print(self.xs.shape)
        print(self.ys.shape)
        quit()
        self.x_data = tor.from_numpy(np.asarray(self.xs,dtype=np.float32)).to(device)
        self.y_data = tor.from_numpy(np.asarray(self.ys,dtype=np.float32)).to(device)

        self.lenx=self.xs.shape[0]
        
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]    
    def __len__(self):
        return self.lenx
    def getnp(self):
        """ Get numpy data """
        return self.xs, self.ys
        
    def chunkea(selfvar,n=15):
        """ redimensiona la serie de tiempo sin repeticion  """
    
        nchunks=var.shape[0]//n
        return var[:nchunks*n].reshape((nchunks,n,var.shape[-2],var.shape[-1]))
    
    def chunks(self,dat,nt_in,nt_out,jvar_in,jvar_out,jcovariates):
        """ Divide data in chunks for recursive NNs 
            Los covariates se asumen conocidos para el forecast
             No repite la entrada.
        """

        dat=sliding_window_view(dat,(nt_in+nt_out,dat.shape[1]))[::nt_in, 0,...]
        #saltos de nt_in para no repetir la entrada
        x=dat[:,:,jvar_in+jcovariates]
        y=dat[:,:,jvar_out]
        #covariates=dat[:,:,jcovariates]
        
#        x=dat[:,:nt_in,jvar_in]
#        y=dat[:,nt_in:,jvar_out]
#        covariates=dat[:,:,jcovariates]
        
        return x,y#,covariates
    
    def normalize_gauss(self,X):
        ''' Normalize to standard Gaussian distribution '''
        self.X_m = np.mean(X, axis = 0)
        self.X_s = np.std(X, axis = 0)   
        return (X-self.X_m)/(self.X_s)
    
    def denormalize_gauss(self,Xnorm):
        ''' Desnormalize Gaussian transformation '''
        return self.X_m + self.X_s * Xnorm

    def normalize_minmax(self,X):
        ''' Normalize to 0,1 interval '''
        self.x_mn = np.min(X, axis = 0)
        self.x_mx = np.max(X, axis = 0)   
        return (X-self.x_mn)/(self.x_mx-self.x_mn)
    
    def denormalize_minmax(self,Xnorm):
        ''' Desnormalize  '''
        return self.x_mn+ (self.x_mx-self.x_mn) * X

    def normalize_minmax_window(self,X):
        ''' Normalize to 0,1 interval per window'''
        self.x_mn = np.min(X, axis=1, keepdims=True)  # [nt, 1, nseries]
        self.x_mx = np.max(X, axis=1, keepdims=True)  # [nt, 1, nseries]
        
        return  (X - self.x_mn) / (self.x_mx - self.x_mn)
    
    def denormalize_minmax_window(self,Xnorm):
        ''' DesNormalize to 0,1 interval per window'''
        return self.x_mn+ (self.x_mx-self.x_mn) * Xnorm

#-------------------------------------------------------------------    
    
if __name__=="__main__":

#    nt=20_000
#    dat_fl=f'./dat/l63/ts-{nt}.npz'
    
#    Mdl=dyn.L63()
#    dat = Mdl.read_ts(dat_fl,nt=nt)
    from read_data import load_ts
    day,date,price,company,volume = load_ts(sectors=['beverages'], assets=['KO','PEP.O'], 
                                        pathdat='dat/')

    Dataset=DriveData(price.T,
                      nt_in=2,
                      nt_out=98,
                      jvar_in=[0],jvar_out=[0], # input and output variables
                      jcovariates=[1], # covariate variables
                      )

    
    
