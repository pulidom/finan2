""" Generate data from a dynamical system or saved data suitable for pytorch 

"""
import sys;  sys.path.insert(0, '../')
import numpy as np
import numpy.random as rnd
from numpy.lib.stride_tricks import sliding_window_view
import torch as tor
from torch.utils.data import Dataset
from torch.utils import data 

def create_dataloaders(ts,conf,dat_type=['train','val','test']):
    """
    Lee o genera los datos y luego carga los data loaders
    """

    
    dat_spec = {'train' : (0,conf.n_train, conf.batch_size, True),
                'val' :  (conf.n_train,conf.n_train+conf.n_val, conf.n_val, False),
                'test' : (conf.n_train+conf.n_val,
                          conf.n_train+2*conf.n_val, conf.n_val, False),}

    ts=ts.T

    nt_out = conf.nt_window
    var_in=[0,1]
    var_out=[0]
    
    loaders = []
    for dat_name in dat_type:
        
        nt0, nt1, batch_size, shuffle = dat_spec[dat_name]

        dat = ts[nt0:nt1]
        
        dset = DriveData(dat, nt_in=1, nt_jump = conf.Njump, nt_out=nt_out, 
                            jvar_in=var_in, jvar_out=var_out)

        loader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
        loaders.append(loader)
    return loaders #loaders[0], loaders[1]

class DriveData(Dataset):
    """ Dada la serie de tiempos la deja lista para utilizar con el DataLoader de pytorch 
    nt_in=1,nt_out=1,  number of time of input and of output variables
     los chunks se eligen cada nt_in tiempos
    """
    def __init__(self,dat, # time series
                 device='cpu',
                 nt_in=1,nt_out=1, # number of time of input and of output
                 nt_jump=100, # jump between windows/intervals
                 jvar_in=[0,2],jvar_out=[0], # input (covariates) and output variables
                 normalize=None,
                 ldeepar=True,
                 ):
        
        if normalize is not None:
            if normalize == 'gauss':                
                dat=self.normalize_gauss(dat)
            elif normalize == 'min-max':
                dat=self.normalize_minmax(dat)
        print('data shape: ',dat.shape)
        self.xs,self.ys = self.chunks(dat,nt_in,nt_out,nt_jump,
                                      jvar_in,jvar_out,
                                      ldeepar=ldeepar)
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
        
    def chunkea(var,n=15):
        """ redimensiona la serie de tiempo sin repeticion  """
    
        nchunks=var.shape[0]//n
        return var[:nchunks*n].reshape((nchunks,n,var.shape[-2],var.shape[-1]))
    
    def chunks(self,dat,nt_in,nt_out,nt_jump,jvar_in,jvar_out,ldeepar=True):
        """ Divide data in chunks for recursive NNs 
            Los covariates se asumen conocidos para el forecast
             No repite la entrada.
        """

        dat=sliding_window_view(dat,(nt_in+nt_out,dat.shape[1]))[::nt_jump, 0,...]
        #saltos de nt_in para no repetir la entrada

        if ldeepar:
            x=dat[:,:-1,jvar_in]
            y=dat[:,1:,jvar_out]
        else:
            x=dat[:,:nt_in,jvar_in]
            y=dat[:,nt_in:,jvar_out]
#        covariates=dat[:,:,jcovariates]
       
        return x,y#,covariates
    
    def normalize_gauss(X):
        ''' Normalize to standard Gaussian distribution '''
        self.X_m = np.mean(X, axis = 0)
        self.X_s = np.std(X, axis = 0)   
        return (X-self.X_m)/(self.X_s)
    
    def desnormalize_gauss(Xnorm):
        ''' Desnormalize Gaussian transformation '''
        return self.X_m + self.X_s * Xnorm

    def normalize_minmax(X):
        ''' Normalize to 0,1 interval '''
        self.x_mn = np.min(X, axis = 0)
        self.x_mx = np.max(X, axis = 0)   
        return (X-self.x_mn)/(self.x_mx-self.x_mn)
    
    def desnormalize_minmax(Xnorm):
        ''' Desnormalize  '''
        return self.x_mn+ (self.x_mx-self.x_mn) * X
    
#-------------------------------------------------------------------    
    
if __name__=="__main__":

    nt=20_000
    dat_fl=f'./dat/l63/ts-{nt}.npz'
    
    Mdl=dyn.L63()
    dat = Mdl.read_ts(dat_fl,nt=nt)
    Dataset=DriveData(dat,
                      nt_in=5,
                      nt_out=10,
                      jvar_in=[0,2],jvar_out=[0], # input and output variables
                      )
