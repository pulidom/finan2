#!/usr/bin/python
"""
Constructor: A linear dynamical system 

x es el estado aumentado= variables y todos los coeficientes
nvar es el numero de variables deel sistema dinamico
"""
import numpy as np
import numpy.random as rnd
import sys
import importlib
from dyn import M
#----------------------------------------------------------
class ldyn(M):
    def __init__( self,dt=0.5, # integration time step
                  nvar=1, #nro de variables del sistema
                  sigma=[0.1], # std parameters for each degree
                  rel_err=0.2, # relative error of the parameters
                  x0=None, # initial mean state
                  var0=None, # initial variance
                  laug=True,
                  M = None, # linear model for fixed parameters
                  **kwargs):
        super().__init__(  dt=dt, **kwargs )

        self.nvar=nvar
        self.laug=laug
        self.nx=nvar**2 if laug else nvar
        self.npar=nvar**2-nvar

        self.M = M
        self.rel_err = rel_err
        # initial guess  mean parameter (for X0)
        if x0 is None:
            x0=np.zeros(self.nx)

        if var0 is None:
            var0=rel_err * np.ones(self.nx)            
        self.x0=x0
        self.var0=var0
        
    #- 3 -------------------------------------------------
    def _mdl(self,x):
        """
            Model equations Up to second order 
                          x[2*nx+nx**2+nx**3]
            x [nx+npar,nem]
        """
        dx=np.zeros(x.shape)
        if self.laug:
            coef = self.stt2coeff(x)
        else:
            coef = self.M
        x_var=x[None,:self.nvar].T
        dx[:self.nvar] = (self.dt *  (coef.T @ x_var)).squeeze().T
        return dx # assumes persistente in the parameters

    def initialization(self,**kwargs): 
        ' perturbations in the parameters relative to the true values (avoid crashing) '
        if self.laug:
            M = np.eye(self.nvar)
            self.x0[self.nvar:] = self.coeff2stt(M)  # transition matrix coef without diagonal 
        X0 = self.x0[:,None] + self.rel_err * (self.var0**.5)[:,None] * rnd.randn(self.nx,self.nem)
        return X0,self.x0

    def coeff2stt(self,M):

        N = M.shape[0]
        mask = ~np.eye(N, dtype=bool)
        enshape=() if M.ndim==2 else (M.shape[2],)
        y=np.zeros(self.nx,)
        if M.ndim == 2:
            # Case: (N, N)
            #y[self.nvar:] = M[mask]
            y = M[mask]
    
        elif M.ndim == 3:
            # Case: (N, N, M)
            #y[self.nvar:] = M[mask, :]  # shape will be (N*N - N, M)
            y = M[mask, :]  # shape will be (N*N - N, M)

        return y
    
    def stt2coeff(self,x):
        
        # Nmero total de coefficientes
        mask = ~np.eye(self.nvar, dtype=bool)

        
        if x.ndim == 1:
            out = np.zeros((self.nvar,self.nvar), dtype=x.dtype)
            out[mask] = x[self.nvar:]

        else: 
            # (N*N - N, M) → (N, N, M)
            out = np.zeros((self.nvar, self.nvar, x.shape[1]), dtype=x.dtype)
            # Use broadcasting: assign values to all matrices at once
            out[mask, :] = x[self.nvar:]            

        return out

        
