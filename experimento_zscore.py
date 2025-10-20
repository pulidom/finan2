import numpy as np
import matplotlib.pyplot as plt
import datetime, os
from read_data import yahoo_download
import arbitrage as ar
import copy
import cointegration as co
# ==========================
# CONFIGURACIÓN BASE
# ==========================
class cnf:
    pathdat = 'dat/'
    tipo = 'log'
    mtd = 'ot'
    Njump = 60
    sigma_co = 1.5
    sigma_ve = 0.2
    nmax = -1
    nsel = 5
    linver_betaweight = 0
    shorten = 0
    fname = 'tmp/pruebas_ventanas/'

os.makedirs(cnf.fname, exist_ok=True)

# ==========================
# DESCARGA DE DATOS
# ==========================
day, date, price, company, volume = yahoo_download(
    ['AR', 'EQT'],
    datetime.date(2015,1,1),
    datetime.date(2025,1,1)
)
print('Descarga completada:', company)

# ==========================
# COMBINACIONES A PROBAR
# ==========================
n_train_vals   = [252*4]
#beta_win_vals  = [60, 120, 252,252*2,252*4]
beta_win_vals  = [int(252/2),252,252*2,252*4]
zscore_win_vals = [int(252/2),252,252*2,252*4]
#[30, 60, 120, 252,252*2,252*4]
#zscore_win_vals = [30]

# ==========================
# LOOP PRINCIPAL
# ==========================
plt.figure(figsize=(8,4))
nt=price.shape[1]
for n_train in n_train_vals:
    for beta_win,zscore_win in zip(beta_win_vals,zscore_win_vals):
        #for zscore_win in zscore_win_vals:
            setattr(cnf, 'Ntraining', n_train)
            setattr(cnf, 'beta_win', beta_win)
            setattr(cnf, 'zscore_win', zscore_win)

            print(f'\nProbando: Ntrain={n_train}, beta={beta_win}, zscore={zscore_win}')
            iini=0
            # correr análisis de pares
            zscore_intime = np.array([])
            for ilast in range(cnf.Ntraining+cnf.Njump,nt,cnf.Njump):
                assets_tr = price[:,iini:ilast]
                volume_tr = volume[:, iini:ilast]
                mean_vol  = np.mean(volume_tr, axis=1)  # promedio por activo    
            
                #t0 = time()
                res_ = ar.all_pairs(assets_tr,company,cnf,)
                res = copy.deepcopy(res_)

                ### Selección de pares según P-val
                #metrics = co.all_pairs_stats(assets_tr[:,:-cnf.Njump],res.company,'asset')
                
                #print(len(metrics.pvalue))
                #exit()
                #print(res.zscore.shape)
                zscore_intime=np.append(zscore_intime,res.zscore[0, -cnf.Njump:])
                #idx = np.argsort(metrics.pvalue)[:cnf.nsel]    

            # graficar zscore promedio (solo para visualización)
            
            plt.plot(zscore_intime, alpha=0.7, label=f'Ventana de {beta_win} días')

            #plt.title(f'Z-Score (Ntrain={n_train}, β={beta_win}, z={zscore_win})')
            plt.xlabel('Tiempo')
            plt.ylabel('Z-Score')
            plt.legend()
            plt.grid(True)

            figpath = f"{cnf.fname}zscore_compar.png"
            plt.tight_layout()
            plt.savefig(figpath)
            #plt.close()
print(f'→ Gráfico guardado en {figpath}')
