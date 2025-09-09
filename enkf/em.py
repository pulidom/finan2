'''   Expectation maximization to estimate Q, R and the (augmented) state
'''
import numpy as np
import numpy.random as rnd
from numpy.linalg import pinv as minv
from scipy.linalg import sqrtm
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression
#from utils import sqrt_svd as sqrtm
from enkf import FILTER as FILT
import utils
#from ldyn import ldyn
import obs

def init_spar(y0,cnf):
    x0 = np.zeros(cnf.Nx)
    x0[:cnf.Ny] = y0
    var0 = np.ones(cnf.Nx)
    var0[:cnf.Ny] = cnf.var0obs * y0**2
    Q0diag        = np.zeros(cnf.Nx)
    Q0diag[:cnf.Nvar] = cnf.Q0[0]
    Q0diag[cnf.Nvar:] = cnf.Q0[1]
    Q0 = np.diag(Q0diag)
    return x0,var0,Q0

def off_diag(x,Nvar):
    ' Transforma a matriz hace 0 la diagonal y retransforma a vector '
    M=x[Nvar:].reshape((Nvar,Nvar))
    np.fill_diagonal(M, 0)
    return M.reshape(-1)
#def set_off_diag(Q0,Nvar):
#    ' Setea los elementos off diagonal y hace 0 la diagonal'
#    Qs=np.ones((Q0.shape[0]-Nvar,Q0.shape[0]-Nvar))
#    np.fill_diagonal(Qs, 0)
#    print(Qs.shape)
#    return Qs
#def init_off_spar(y0,cnf):
#    ''' Setea los parametros con los elementos de la diagonal de M = 0
#    '''
#    x0 = np.zeros(cnf.Nx)
#    x0[:cnf.Ny] = y0
#    var0 = np.ones(cnf.Nx)
#    var0[:cnf.Ny] = cnf.var0obs * y0**2
#    var0[cnf.Nvar:] = off_diag(var0,cnf.Nvar)
#    Q0diag        = np.zeros(cnf.Nx)
#    Q0diag[:cnf.Nvar] = cnf.Q0[0]
#    Q0 = np.diag(Q0diag)
#    Q0[cnf.Nvar:,cnf.Nvar:] = cnf.Q0[1] * set_off_diag(Q0,cnf.Nvar)
#    return x0,var0,Q0

def exp_filter(o_kwargs,cnf,y_t=None, x0=None, var0=None, Q0=None, sfilter="filter"):
    ' Solo usa el filtro sin el EM'
    Class_Filter = {
        "filter": FILTER,
        "lfilter": LFILTER,
    }        

    #laug=True if sfilter=="filter" else False
    laug=True
    
    Tmdl = cnf.ldyn(dtcy=cnf.dtcy,nem=cnf.Nem,x0 = x0, var0=var0,
                      nvar= cnf.Nvar,rel_err=cnf.rel_error,
                      laug=laug)
    
    X0,xt0 = Tmdl.initialization()
   
    print('Generating observations')
    o_kwargs = utils.obj2dict(o_kwargs)
    Obs = obs.OBS(**o_kwargs)
    
    Filter=Class_Filter[sfilter](Tmdl,Obs,assmtd='perobs',
                  finf=cnf.finf,Q0=Q0,llast=cnf.llast,
                  niter=cnf.niter_first)
    
    Xa_t,Xf_t = Filter.filter(X0,y_t) # hago predicciones y analisis

    #res,rmse=Filter(X0,y_t)
    
    return Xa_t,Xf_t

def exp_iniem(o_kwargs,cnf,y_t=None, x0=None, var0=None, Q0=None, sfilter="filter"):
    ' Initial EM without predictions'
    Class_Filter = {
        "filter": FILTER,
        "lfilter": LFILTER,
    }        

    #laug=True if sfilter=="filter" else False
    laug=True
    
    Tmdl = cnf.ldyn(dtcy=cnf.dtcy,nem=cnf.Nem,x0 = x0, var0=var0,
                      nvar= cnf.Nvar,rel_err=cnf.rel_error,
                      laug=laug)
    
    X0,xt0 = Tmdl.initialization()
   
    print('Generating observations')
    o_kwargs = utils.obj2dict(o_kwargs)
    Obs = obs.OBS(**o_kwargs)
    
    Filter=Class_Filter[sfilter](Tmdl,Obs,assmtd='perobs',
                                 finf=cnf.finf,Q0=Q0,llast=cnf.llast, lRest=1 ,
                                 niter=cnf.niter_first)

    res,rmse=Filter(X0,y_t)
    
    return res,rmse,Filter


class FILTER(FILT):
    def __init__(self,*args, niter=10,
                 lQest=1, lRest=0 , lrmse=1, llast=0, Q0=None,
                 **kwargs):
        super().__init__(*args,**kwargs) # inicializo con los atributos del filtro
        self.niter=niter
        self.lQest=lQest
        self.lRest=lRest
        self.llast=llast
        if Q0 is not None and lQest==1:
            self.sqQ = sqrtm(Q0)
        else:
            #self.sqQ =np.zeros((6,6))
            print('Q estimation requiries Q initialization')

            
    #--------------------------------------------
    def filter(self,Xf,y_t):
        """
            assimilation cyle.
            Observations are expected at it=1 
            and Xf is at it=1
            Xa_t, Xf_t Outputs are all at it=1
           (repite el filter del EnKF pero con integracion
           en diferente lugar para el EM)
        """

        [nx, nem] = Xf.shape; [ncy, ny] = y_t.shape
        Xa_t,Xf_t=np.zeros((2,ncy,nx,nem))

        for icy in range(ncy):
            
            # Evolve forecast ensemble members
            if icy > 0: Xf = self.integ_noise(Xa)
            
            Xa=self.assmtd(Xf,y_t[icy])
            
            Xa_t[icy]=Xa
            Xf_t[icy]=Xf
        
        return Xa_t,Xf_t

    #--------------------------------------------
    def integ_noise(self,X0):
        ' Add noise after evolving the model '
        Xf  = self.integ(X0)
        Xf += self.sqQ @ rnd.randn(*Xf.shape)
        Xf = Xf + (self.finf-1)[:,None] * (Xf-Xf.mean(1,keepdims=True))
        return Xf
    
    #---------------------------------------------
    def em(self,X0,y_t):
        """
            EM for statistical parameter estimation
            Observations are expected at it=1 (xa is at it=0)
            X0 is expected to be an augmented state with the parameters
        """
        
        [nx, nem] = X0.shape
        [ncy, ny] = y_t.shape

        rmse=np.zeros((ncy-1,3))
        res=[]
        # Evolve forecast ensemble members up to it=1 (first observatoin
        Xf = self.integ(X0)

        for iter in range(self.niter):
            Xa_t,Xf_t = self.filter(Xf,y_t)
            Xs_t = self.enrts( Xf_t, Xa_t  )
            
            if self.lQest: self.estimate_Q(Xs_t)
            if self.lRest: self.estimate_R(Xs_t,y_t)

            if self.llast: # uso el ultimo parametro
                Xf[:3]=Xs_t[0,:3]
                Xf[3:]=Xs_t[-1,3:]
                # spread similar al Xs
                #Xf[3:]=(Xs_t[-1,3:].mean(-1,keepdims=True)+
                #        (Xs_t[0,:3]-Xs_t[0,:3].mean(-1,keepdims=True)))
            else:
                Xf=Xs_t[0]

            #print('Despues de smoothing: ',Xf[:ny,:].mean(-1))
            
            rmse[iter] = self.diagnostic(Xf_t,Xa_t,Xs_t, y_t)
            print(iter,'Lik: ',rmse[iter,2])
            #print('RMSE: ', self.diagnostic2(Xf_t,Xa_t,Xs_t, y_t))
            #if rmse[iter,2]> -4000:
            #    res.append({'Xa':Xa_t,'Xs':Xs_t,'Xf':Xf_t,
            #               'sqQ':self.sqQ})#
            #    print('DETENGO por LIK')
            #    break
            if iter==0 or iter==self.niter-1: # first and last iteration
                res.append({'Xa':Xa_t,'Xs':Xs_t,'Xf':Xf_t,
                           'sqQ':self.sqQ})#
                
            
        #print('Qvar: ',np.diag((self.sqQ @ self.sqQ)[:5,:5]) )
        #print('Qpar: ',np.diag((self.sqQ @ self.sqQ)[5:,5:]) )

        return res,rmse

    __call__=em
    #--------------------------------------------
    def filter_alone(self,Xa,y_t):
        Xf = self.integ_noise(Xa)
        Xa_t,Xf_t = self.filter(Xf,y_t)        
        return Xa_t,Xf_t
        
    #--------------------------------------------
    def estimate_Q(self,Xs_t):
        
        ncy,nx,nem = Xs_t.shape
        
        Q=np.zeros([nx,nx])
            
        for it in range(1,ncy):
            Xf = self.integ(Xs_t[it-1]) # without model error
            Xf = Xf + (self.finf-1)[:,None] * (Xf-Xf.mean(1,keepdims=True))
            Z = Xs_t[it] - Xf
            Q += (Z @ Z.T)

        Q /= ((ncy-1)*nem) # nem-1?
        #Q /= ((ncy-1)*(nem-1)) # nem-1?

        self.sqQ=sqrtm(Q) # actualiza

    #--------------------------------------------
    def estimate_R(self,Xs_t,y_t):
        
        ncy,ny = y_t.shape
        nem = Xs_t.shape[-1]
        
        R=np.zeros([ny,ny])
            
        for it in range(1,ncy):
            
            Z = y_t[it][:,None] - self.Hop(Xs_t[it])
            
            R+= (Z @ Z.T)

        #R /= ((ncy-1)*nem) # nem-1?
        R /= ((ncy-1)*(nem-1)) # nem-1?

        self.sqR=sqrtm(R)
        self.R=R
    #--------------------------------------------
    def diagnostic(self,Xf_t,Xa_t,Xs_t, y_t):

        if self.xt_t is not None:
            rmse= np.array ([
                utils.rms(Xa_t.mean(2)-self.xt_t),# estado aumentado?
                utils.rms(Xs_t.mean(2)-self.xt_t),
                self.likelihood(Xf_t, y_t)
            ])
        else:  
            rmse= np.array ([
                self.likelihood(Xf_t, y_t)
            ])
          
        return rmse
    
    #--------------------------------------------
    def diagnostic2(self,Xf_t,Xa_t,Xs_t, y_t):

        rmse= np.array ([
            utils.rms(Xa_t.mean(2)[...,:3]-self.xt_t[...,:3]),# estado
            utils.rms(Xs_t.mean(2)[...,:3]-self.xt_t[...,:3]),
            utils.rms(Xa_t.mean(2)[...,3:]-self.xt_t[...,3:]),# parametros
            utils.rms(Xs_t.mean(2)[...,3:]-self.xt_t[...,3:]),
            np.sqrt(((Xa_t.mean(2)[-1,3:]-self.xt_t[-1,3:])**2).mean()), # last parameters
            self.likelihood(Xf_t, y_t)
        ])
        return rmse


def predict(x,M,n_steps=1):
    forecast = []
    x_last=np.copy(x)
    
    for i in range(n_steps):
        # Predict next state: new_state = transition_matrix * last_state
        x_last =  x_last @ M
        forecast.append(x_last)

    # Convert forecast to a numpy array for easier manipulation
    return np.array(forecast)

class LFILTER(FILTER):
    '''   Linear EM uses traditional KF from pykalman and linear regression
    ''' 
    def __init__(self,*args, niter=10, llast=0, Q0=None,
                 **kwargs):
        super().__init__(*args,**kwargs) # inicializo con los atributos del filtro
        self.niter = niter
    def em(self,Xf,y_t):

        nt,ny = y_t.shape
        nx=Xf.shape[0]
        
        variance_unobs=4# ???
        
        # Si una variable no es observada le agrego ruido blanco como observacion        
        yidx = self.H @ np.arange(1,nx+1)
        missing_indices = np.setdiff1d(np.arange(nx), yidx-1)
        yx_t=np.zeros((nt,nx))
        yx_t[:,(yidx-1).astype(int)]=y_t
        yx_t[:,missing_indices]=rnd.normal(0,variance_unobs,(nt,len(missing_indices))) 
        inputs = yx_t[:-1]
        targets = yx_t[1:]

        # initializations
        x_b = np.zeros(nx) 
        B   = np.eye(nx)
        loglik = []

        print('entra a la iteracion: ')
        for i in range(self.niter):

            print('##### Iteration ' + str(i) + ' #####')

            # Kalman parameters
            reg   = LinearRegression(fit_intercept=False).fit(inputs, targets)

            #bias = reg.intercept_ # if you include the intercept
            M     = reg.coef_
            Q     = np.cov((targets - reg.predict(inputs)).T)


            # apply the Kalman smoother
            kalman = KalmanFilter(#transition_offsets = bias, # if you include the intercept
                                        transition_matrices = M, observation_matrices = self.H,
                                        initial_state_mean = x_b, initial_state_covariance = B,
                                        transition_covariance = Q, observation_covariance = self.R)
            x_s, P_s = kalman.smooth(y_t)

            x_b = x_s[0,:]
            B   = P_s[0,:,:]

            perturb = np.zeros(x_s.shape)
            for t in range(nt):
                perturb[t] = np.random.multivariate_normal(x_s[t], P_s[t])

            inputs = np.copy(perturb[0:-1])
            targets = np.copy(perturb[1:]) 

            llik = kalman.loglikelihood(y_t)
            print( i, llik )

            loglik.append( llik )

        x_f = predict(x_s,M, n_steps=5)

        self.sqQ = sqrtm(Q) # actualizo para el filtro
        
        res=[0] # mimic EM
        res.append({'loglik':loglik,'x_s':x_s,'Ps':P_s,'M':M,
                    'Xf':x_f,'Xa':x_s})
        
        return res,0
    __call__=em
