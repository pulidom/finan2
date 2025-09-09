#!/usr/bin/python3
#
# Ensemble kalman filter 
#    includes perobs, etkf, whitaker, extkf, enRTS, LETKF
#  
# load python modules
#import sys;  sys.path.insert(0, '../')
import numpy as np
from numpy.linalg import pinv
import numpy.random as rnd
import utils
from utils import PaWa_svd as PaWa #using svd

class FILTER:
    def __init__(self,Fmdl,Obs,
                 assmtd='perobs',finf=1.0,nlocx=-1,
                 finfpar=None, # inflacion en los parametros
                 xt_t=None,
                 ):


        self.Hop=Obs.Hop
        self.H=Obs.H
        self.R=Obs.sqR @ Obs.sqR
        self.sqR = Obs.sqR
        self.xt_t=xt_t # true state for diagnostics

        self.integ=Fmdl.integ
        if assmtd=='perobs' and Obs.swH != 0:
            raise SystemExit('pert obs only works for linear H')

        self.assmtd=eval('self.'+assmtd)
        self.finf=finf
        self.nlocx=nlocx #localization range/order

    #--------------------------------------------
    def asscy(self,Xa,y_t):
        """
            assimilation cyle.
            Observations are expected at it=1 (xa is at it=0)
            Xa_t, Xf_t Outputs are all at it=1
        """

        [nx, nem] = Xa.shape; [ncy, ny] = y_t.shape
        Xa_t,Xf_t=np.zeros((2,ncy,nx,nem))

        print ('Assimilating ...')
        # assimilation cycles
        for icy in range(ncy):
            
            # Evolve forecast ensemble members
            Xf = self.integ(Xa)
            # Assimilate
            Xa=self.assmtd(Xf,y_t[icy])
            
            # save time dependent variables
            Xa_t[icy]=Xa
            Xf_t[icy]=Xf
        return Xa_t,Xf_t

    __call__=asscy
    
    #--------------------------------------------
    def perobs(self,Xf,y):
        " Assimilate using classical enkf. Pertubed observations "
        [nx,nem]=Xf.shape; ny=y.shape[0]

        H=self.H
        Pfxx= np.atleast_2d(np.cov(Xf))
        # Classical Kalman gain
        K = (Pfxx @ H.T) @ pinv(H @ (Pfxx @ H.T) + self.R)
        wrk=rnd.normal(0, 1, [ny,nem])
        Y=H @ Xf+self.sqR @ wrk
        
        Xa = Xf + K @ (y[:,None] - Y)
            
        return Xa

    
    #--------------------------------------------
    def perobs2(self,Xf,y):
        """ Observaciones perturbadas  EnKF
             Evita calculo de la covarianza
             Usa diferencias del operador observacional 
                (suitable para nonlinear map)
             Xf [Nx,Nem],y [Ny]
             implicit: H, sqR, R
             \[ X^a = X^f + X'  [ \mathcal{H}'(x) ]
                 [ \mathcal{H}'(x)^\tau \mathcal{H}'(x) + R ]^{-1} 
                 ( y- \mathcal{H}(x) ) \]
        """

        
        nem=Xf.shape[1]; ny=y.shape[0]
        Hx = self.Hop(Xf) # # Observational map Ny,Nem
        
        xfm,Xper = meanper(Xf)        
        hxm,Yper = meanper(Hx)

        D = Yper @ Yper.T + self.R * (nem-1)

        K = Xper @ mrdiv(Yper.T,D) 
        Noise = self.sqR @ rnd.normal(0,1,[ny,nem])
        Xa = Xf + K @ (y[:,None] - Hx + Noise)

        # Alternativa colapsar al ny? primero
        #Noise = self.sqR @ rnd.normal(0,1,[ny,nem])
        #w = self.mrdiv(Yper.T,D) @ (y[...,np.newaxis] - Hx + Noise)
        #Xa = Xf + Xper @ w
        
        return Xa
    
    #--------------------------------------------
    def enrts( self,Xf_t, Xa_t ):
        """ Ensemble RTS smoother Cosme et al 2012
            Asumo que tengo observacion en it=0 y que tengo un 
            analisis en it=0 (las CIs son en "it=-1")
            Luego evoluciono el modelo a it=0 y ahi comienzo asimilacion
            Forecasts are at  the times of the index
 
        """

        ncy, nx, nem,  = Xf_t.shape
        Xs_t = np.zeros([ncy, nx, nem])

        Xs_t[-1] = Xa_t[-1] # initialization of the smoother

        for it in range(ncy-2,-1,-1):
            #    Paf = np.cov(Xa_t[:,:,it], Xf_t[:,:,it+1])[:nx, nx:]
            Paf  = utils.cova(Xa_t[it], Xf_t[it+1])
            Pff = np.atleast_2d(np.cov(Xf_t[it+1]))

            K = Paf @ pinv(Pff)
            
            Xs_t[it] = Xa_t[it] + K @ (Xs_t[it+1] - Xf_t[it+1])

        return Xs_t
    
    #--------------------------------------------
    def etkf(self,Xf,yo):
        " Assimilate using Hunt et al, etkf, square root  enkf"
        #    from com.com.PaWa_eig as PaWa #using eig
        if self.finf < 0: finf=1
        else: finf=self.finf
        
        # implicit inputs
        Rinv=pinv(self.sqR @ self.sqR.T)
        nem = Xf.shape[1]
        #--------------------
        
        Yf = self.Hop(Xf) # Observational map

        # mean and perturbations
        xfm,Xper = utils.meanper(Xf)        
        hxfm,Yper = utils.meanper(Yf)

        C = Yper.T @ Rinv
        B = (nem-1.)/finf*np.eye(nem)+ C @ Yper

        Pa , Wa = PaWa ( B ) # inverse and square root via SVD or EIG

        innov = yo[...,np.newaxis] - hxfm
        wa = Pa @ (C @ innov)

        Xa = xfm + Xper @ (wa+Wa)

        return Xa
  
    #--------------------------------------------
    def likelihood(self, Xf_t, y_t):
        "innovation log likelihood"
    
        ncy,ny = y_t.shape        
        xmean = Xf_t.mean(2)

        l = 0
        for it in range(ncy):

            hxf = np.cov(self.Hop(Xf_t[it])) + self.R            
            hxfdet = np.linalg.det(hxf)

            d  = y_t[it] - self.Hop( xmean[it] )
            d1 = pinv(hxf) @ d
            l -= .5 * np.log(hxfdet)
            l -= .5 * d.T @ d1

        return l

