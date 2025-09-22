#!/usr/bin/python
"""
Initialization and integration of a dynamical system

Lorenz-63 
Lorenz-96 

[2025-01-09] Simplicacion codigos. Uso herencia.
 
Author Manuel Pulido


 Reference:

Pulido M., G. Scheffler, J. Ruiz, M. Lucini and P. Tandeo, 2016: Estimation of the functional form of subgrid-scale schemes using ensemble-based data assimilation: a simple model experiment. Q. J.  Roy. Meteorol. Soc.,  142, 2974-2984.

http://doi.org/10.1002/qj.2879


"""
import sys; sys.path.insert(0, '../')
import os
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import enkf.utils_enkf as utils

#----------------------------------------------------------
class M: # parent dynamical model class
    def __init__(self,
                 dtcy=0.01, # output times
                 nem=100, # ensemble size
                 dt = None, # integration time step
                 ):
        self.dtcy=dtcy        
        self.kt= int(dtcy/dt) # number of model time steps between observations
                              #(outputs of the model) 
        self.dt=dtcy/self.kt # exact integration time step
        
        self.nem=nem
        self.x0=0 # mean IC for initialization
        self.x0std=1 # and its std dev
        
    def __call__(self,xold):
        return self.integ(xold)
    
    def integ(self,xold):
        " Integrate in a dtcy time "

        x=np.copy(xold)
        for it in range(self.kt):
            x = self.__rk4( x )

        return x
    
    def __rk4(self,xold):
        " 4th order Runge Kutta "

        dx1 = self._mdl( xold )
        dx2 = self._mdl( xold + 0.5 * dx1 )
        dx3 = self._mdl( xold + 0.5 * dx2 )
        dx4 = self._mdl( xold + dx3 )

        x = xold +  ( dx1 + 2.0 * (dx2 + dx3) + dx4 ) / 6.0
        return x

    def initialization(self,nt_spinup = 7200, nt_corr = 30,
                       seed = 25,
                       nt_samples = 1000):
        " Set single or ensemble initial conditions from a climatology"

        print('Generating initial conditions')
        
        # Spinup integration
        rnd.seed(seed)
        x  = self.x0 + self.x0std * rnd.normal(0, 1, self.nx) # randon initial condition

        kt_orig  = self.kt
        self.kt=kt_orig*nt_spinup        
        x  = self.integ(x) # full spinup 

        # Climatological integration with uncorrelated states
        self.kt = nt_corr*kt_orig # uncorrelated states
        
        print('Climatology')
        x_t = np.zeros((nt_samples,self.nx))
        x_t[0]=x
        for it in range(1,nt_samples):
            x_t[it] = self.integ(x_t[it-1]) # trajectory (climatology)
            
        self.kt = kt_orig
        
        #  Select randomly initial condition and  initial ensemble states        
        # reset random seed
        rnd.seed(seed+1) #to choose always the sampe xt (after clim) and initial ensemble
        
        idx = np.random.choice(nt_samples,self.nem+1,replace=True)            
        print ( 'Take initial condition from climatology' )

        X0 = x_t[idx[:self.nem]]
        xt0 = x_t[idx[self.nem]]
        
        return xt0,X0.T,
        
#----------------------------------------------------------
class L63(M):
    def __init__( self,dt=0.005, # integration time step
                  sigma=10, rho=28., beta=8/3., # model paramaters
                  **kwargs):
        super().__init__(  dt=dt, **kwargs )
        
        """ Initializes L63 params
              background parameters sigma=11.5,beta=2.87,rho=32
        """        
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.nx = 3
        
    def _mdl(self,x):
        " L63 model equations "

        dx0 = self.sigma * (x[1] - x[0])
        dx1 = x[0] * (self.rho - x[2]) - x[1]
        dx2 = x[0] * x[1] - self.beta * x[2]
        
        return self.dt * np.array([dx0,dx1,dx2])
    
'''    def initialize(self):
        ' Starting from a known value the climatology'
        self.x0 = np.array([6.39435776, 9.23172442, 19.15323224])
        self.x0std=0 # and its std dev
        X0,xt = M.initialization(self,nt_spinup=0)
        
        return X0,xt 
'''
    
    
#----------------------------------------------------------
class LV(M):
    def __init__( self,dt=0.05, # integration time step
                  alpha=0.3, beta=0.9, gamma=0.5, delta=0.4, # model
#                  alpha=0.5, beta=0.7, gamma=0.35, delta=0.35, # model paramaters
                  **kwargs):
        super().__init__(  dt=dt, **kwargs )
        
        """ Lotka Volterra from Lipnick et al 
        """        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.nx = 2
        
    def _mdl(self,x):
        " LV model equations "

        dx0 = self.alpha * x[0] - self.beta * x[0] * x[1]
        dx1 = - self.gamma * x[1] + self.delta * x[0] * x[1]
        
        return self.dt * np.array([dx0,dx1])
   
#----------------------------------------------------------
class augLV(M):
    def __init__( self,dt=0.05, # integration time step
                  alpha=0.3, beta=0.9, gamma=0.5, delta=0.4, # model
#                  alpha=0.5, beta=0.7, gamma=0.35, delta=0.35, # model paramaters
                  **kwargs):
        super().__init__(  dt=dt, **kwargs )
        
        """ Lotka Volterra from Lipnick et al 
             con los parametros alpha y gamma unknown
        """        
        self.alpha = alpha # unknown
        self.beta = beta
        self.gamma = gamma # unknown
        self.delta = delta
        self.nx = 4
        
        # Integration  without parameter perturbations
        self.x0=np.zeros(self.nx)
        self.x0[2:]=[alpha,gamma]  # true parameters
#        self.x0std=np.zeros(self.nx)
#        self.x0std[:3]=[1,1]
        
    def _mdl(self,x):
        " LV model equations "

        dx=np.zeros(x.shape)
        dx[0] = x[2] * x[0] - self.beta * x[0] * x[1]
        dx[1] = - x[3] * x[1] + self.delta * x[0] * x[1]
        
        return self.dt * dx # persistence for the parameters    

    
#------------------------------------- ---------------------
class augL63(M):
    ''' Augmented Lorenz 63 with the parameters in the state vector 
    '''
    def __init__( self, dt=0.005, # integration time step
                  sigma=10, rho=28., beta=8/3., # model paramaters
                  par0=None, # initial mean value of the ensemble parameters
                  rel_err=0.2, # relative error of the parameters 
                  **kwargs):
        super().__init__( dt=dt, **kwargs )


        self.nx=6

        # for initialization (parent class)
        # Integration  without parameter perturbations
        self.x0=np.zeros(self.nx)
        true_params =[sigma,rho,beta] 
        self.x0[3:]=true_params # true parameters
        self.x0std=np.zeros(self.nx)
        self.x0std[:3]=[1,1,1]
         
        # initial guess  mean parameter (for X0)
        if par0 is None:
            par0 = true_params
            par0 = par0 * (1 + rel_err * rnd.normal(0,1,3))
        self.par0=par0
        self.rel_err=rel_err
        
    def _mdl(self,x):
        " augmented L63 model equations [assume parameter persistence]"
        
        dx=np.zeros(x.shape)
        dx[0] = x[3] * (x[1] - x[0])
        dx[1] = x[0] * (x[4] - x[2]) - x[1]
        dx[2] = x[0] * x[1] - x[5] * x[2]

        return self.dt * dx
    
        
    def initialization(self,**kwargs): 
        ' perturbations in the parameters relative to the true values (avoid crashing) '
        X0,xt = M.initialization(self,**kwargs)

        wrk = rnd.normal(0, 1, [3,self.nem])
        #X0[3:,:] =  self.par0[...,None] * (1 + self.rel_err * wrk)
        X0[3:,:] =  self.par0[...,None] + xt[3:,None] * self.rel_err * wrk
        return X0,xt
    
#----------------------------------------------------------
class L96(M):
    '''  
    Lorenz-96 dynamical system 
    \[ d_t X_k = - X_{k-1} (X_{k-2} - X_{k+1} ) - X_k + F \]

    Uses list of indices to speed up calculations
    Works with ensembles (second dimension) 
    '''
    def __init__( self, dt=0.01, F=8., nx=40, **kwargs):
        super().__init__(  dt=dt, **kwargs)

        self.dt=dt
        self.nx=nx
        self.F = F 
        
        self.indm2=list(range(-2,nx-2))
        self.indm1=list(range(-1,nx-1))
        self.indp1=list(np.arange(1,nx+1)%nx)

    def _mdl(self,x):
        " L96 model equations Requires nx as first dim x[nx,npa/nem]"
        dx = (x[self.indp1] - x[self.indm2]) * x[self.indm1] - x + self.F
        return dx * self.dt
    
    

#----------------------------------------------------------
if __name__ == '__main__':
    
    l63 = L63(dtcy=0.01)
    X0,x0 = l63.initialization()
    #
    x_t[0]=x0
    x_t=np.zeros([1000,xold.shape[0]])
    for it in range(1,1000):
        x_t[it]=l63(x_t[it-1])

#    import com.assplots as plotx_t[0,:]

    plt.plot(x_t.T[0])
    plt.plot(x_t.T[1])
    plt.plot(x_t.T[2])
#    ax=plt.figure().add_subplot(projection="3d")
#    ax.plot(x_t[0,:],x_t[1,:],x_t[2,:])
#    plt.savefig('tmp/l63new.eps')
