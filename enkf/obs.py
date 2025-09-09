#!/usr/bin/python
"""  observational mapping
 
      Define H operator and its adjoint (grad H).
      Define sqR 
      Generate synthetic observations

[2025-01-09] Simplicacion codigos.
 
Author Manuel Pulido


 Reference:

Pulido M., G. Scheffler, J. Ruiz, M. Lucini and P. Tandeo, 2016: Estimation of the functional form of subgrid-scale schemes using ensemble-based data assimilation: a simple model experiment. Q. J.  Roy. Meteorol. Soc.,  142, 2974-2984.

http://doi.org/10.1002/qj.2879
"""
# load python modules
import numpy as np
import numpy.random as rnd
from  numpy.linalg import pinv
from scipy.linalg import sqrtm

class OBS:
    def __init__(self,
                 H=None, # observation matrix [ny,nx] 
                 R=None, # observational error covariance [ny,ny]
                 swH=0,  # Type of operator: linear, quadratic, abs, log
                 ):

        self.H=H
        self.Hop,self.Htop=self.set_Hop(swH,H)
        self.ny=H.shape[0]
        self.R = R
        self.sqR = sqrtm(R)
        self.invR = pinv(R)
        self.swH=swH
        
    #- 3 --------------------------------------------------
    def generate_obs( self,x, # initial state  (true)
                      ncy,  # number of cycles/observation times
                      Mdl, # true model   
                     ):
        
        # state dimension; obs
        nx=np.shape(x)[0]; ny=self.ny 

        y_t=np.zeros((ncy,ny))
        x_t=np.zeros((ncy,nx))
            
        # assimilation cycles
        for it in range(ncy):
            
            # Evolve truth
            x = Mdl.integ(x) #
            x_t[it] = x # the initial condtion is not included
            
            # Map observation space and add observation error
            y_t[it] = self.Hop(x) + self.sqR @ rnd.normal(0, 1, ny) 
            
        return x_t, y_t
    
    __call__=generate_obs
    
    def set_Hop(self,swH,H):
        """ Observational operator and its adjoint """
        
        if swH==0: # linear H
            return lambda x: H @ x, lambda x,y: H.T @ y
        elif swH==1: # quadratic H
            return  lambda x: H @ (x**2), lambda x,y: 2 * x * (H.T @ y)
        elif swH==2: # absolute value
            return lambda x: np.abs(x), lambda x,y: np.sign(x) * (H.T @ y)
        elif swH==3: # log
            epsilon=1.e-9
            x_safe = np.clip(np.abs(x), epsilon, None) * np.sign(x)
            return  lambda x: H @ np.log(np.abs(x_safe)), lambda x,y: 1/x_safe * (H.T @ y)
