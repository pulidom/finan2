### Para distintas configuraciones de hiperparametros me guardo un df
### Con los movimientos por dia, para las mejores 5,10 y 20 selecciones
### El capital acumulado a lo largo de los dias, El sharpe calculado mediante
### la funcion de cointegration (sharpe_co) y el Sharpe Ratio calculado como 
### la media de los retornos sobre el desvio (sharpe) (sin anualizar aun, faltaria multiplicar por sqrt(252))
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import multiprocess
import pandas as pd
import psutil
from itertools import product

from read_data import load_ts
import arbitrage as ar
import cointegration as co

# Configuración
class cnf:
    pathdat = 'dat/'
    tipo = 'asset'
    mtd = 'on'
    beta_win = 121
    zscore_win = 41
    nmax = 20
    nsel = 20
    fname = '18_07/prueba_all'
    linver_betaweight = 0
    industry = 'marine'
    shorten = 0
    ncores = psutil.cpu_count(logical=False)
#industrias = ["biotechnology","machinery","media"]
#industrias = ["biotechnology","machinery","media","metals","oil","airlines","construction","defense","insurance",
#              "semiconductors","software","pharmaceuticals","chemicals","capital","water",
#              "beverages","media"] # + marine
os.makedirs(cnf.fname, exist_ok=True)
# Listas de hiperparámetros
#Ntraining_l = [121, 131, 141, 151, 161]
Ntraining_l = [121,131]

#Njump_l = [60, 70, 80, 90]
Njump_l = [80]

#sigma_co_l = [0.5 ,1.0 ,1.5 ,2.0 ,2.5]
sigma_co_l = [1.5]
sigma_ve_l = [0.1]
#sigma_ve_l = [0.0, 0.1 ,0.2, 0.5 ,1.0]
# Cargar datos una sola vez
day, date, price, company = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)
print(day.shape,date.shape,price.shape,company.shape)
print(type(price),type(company))
#for industria in industrias:
#    print("opening data from: ",industria)
#    day_var,date_var,price_var,company_var = load_ts(sector=industria,pathdat=cnf.pathdat)

#    if not np.array_equal(day, day_var):
#        print("WARNING: 'day' y 'day_var' son diferentes.")
#    else:
#        print("'day' y 'day_var' son iguales.")

#    if not np.array_equal(date, date_var):
#        print("WARNING: 'date' y 'date_var' son diferentes.")
#    else:
#        print("'date' y 'date_var' son iguales.")
#    print(price_var.shape,company_var.shape)
#    price = np.concatenate((price, price_var), axis=0)
#    company = np.concatenate((company,company_var),axis=0)
#    print(price.shape,company.shape)
#print("Todas las variables fueron abiertas")


class EvaluateHyperParams:
    def __init__(self, price, company, cnf):
        self.price = price
        self.company = company
        self.cnf = cnf
        self.nt = price.shape[1]

    def __call__(self, params):
        Ntraining_val, Njump_val, sigma_co_val, sigma_ve_val = params
        try:
            print(f"\n⏳ Ejecutando Ntraining={Ntraining_val}, Njump={Njump_val}, sigma_co={sigma_co_val}, sigma_ve={sigma_ve_val}")
            cnf_copy = copy.deepcopy(self.cnf)
            cnf_copy.Ntraining = Ntraining_val
            cnf_copy.Njump = Njump_val
            cnf_copy.sigma_co = sigma_co_val
            cnf_copy.sigma_ve = sigma_ve_val

            caps = [[] for _ in range(3)]
            iini = 0

            for ilast in range(cnf_copy.Ntraining + cnf_copy.Njump, self.nt, cnf_copy.Njump):
                print(f" Ventana: {iini} ➝ {ilast} | Longitud: {ilast - iini}")

                assets_tr = self.price[:cnf_copy.nmax, iini:ilast]
                iini += cnf_copy.Njump

                res = ar.all_pairs(assets_tr, self.company[:cnf_copy.nmax], cnf_copy)
                metrics = co.all_pairs_stats(assets_tr[:, :ilast - cnf_copy.Njump], self.company, cnf_copy.tipo)
                idx = np.argsort(metrics.half_life)[:cnf_copy.nsel]
                res.reorder(idx)

                caps[0].append(res.retorno[:5, cnf_copy.Ntraining:].mean(0))
                caps[1].append(res.retorno[:10, cnf_copy.Ntraining:].mean(0))
                caps[2].append(res.retorno[:20, cnf_copy.Ntraining:].mean(0))

            rets = np.array([np.concatenate(cap) for cap in caps])
            print(len(rets))
            caps_array = np.zeros_like(rets)
            caps_array[:, 0] = 100
            for i in range(3):
                caps_array[i, 1:] = caps_array[i, 0] * np.cumprod(1 + rets[i, 1:])
            print(caps_array[0,0:5])
            print(caps_array[0,-5:-1])
            final_5 = caps_array[0, -1]
            final_10 = caps_array[1, -1]
            final_20 = caps_array[2, -1]
            rets_5 = rets[0]
            print(rets_5[0:5])
            print(rets_5[-5:-1])
            exit()
            rets_10 = rets[1]
            rets_20 = rets[2]
            sharpe_5 = np.mean(rets_5) / np.std(rets_5) # a estos para graficar los multiplique por sqrt(252) para normalizar por año
            sharpe_10 = np.mean(rets_10) / np.std(rets_10) # a estos para graficar los multiplique por sqrt(252) para normalizar por año
            sharpe_20 = np.mean(rets_20) / np.std(rets_20) # a estos para graficar los multiplique por sqrt(252) para normalizar por año
            median_5 = np.median(rets_5)
            median_10 = np.median(rets_10)
            median_20 = np.median(rets_20)
            print("before calculo de sharpe")
            print(type(caps_array),caps_array.shape)
            print(caps_array[0,0:5],caps_array[0,-5:-1])
            print(caps_array[1,0:5],caps_array[1,-5:-1])
            sharpe_5_co = co.sharpe_ratio(caps_array[0])                  
            sharpe_10_co = co.sharpe_ratio(caps_array[1])                  
            sharpe_20_co = co.sharpe_ratio(caps_array[2])                  
            print("after calculo de sharpe", np.sqrt(252)*sharpe_5,sharpe_5_co,np.sqrt(252)*sharpe_10,sharpe_10_co,np.sqrt(252)*sharpe_20,sharpe_20_co)

            print(f"✅ Final: Ntrain={Ntraining_val}, Njump={Njump_val}, sigma_co={sigma_co_val}, sigma_ve={sigma_ve_val} | Top5={final_5:.2f} | Top10={final_10:.2f} | Top20={final_20:.2f}")
            print("sharpe_5: ",sharpe_5,"sharpe_10: ",sharpe_10,"sharpe_20: ",sharpe_20,"median_5: ",median_5,"median_10: ",median_10,"median_20: ",median_20)
            print('len rets 5',len(rets_5),'len rets 10',len(rets_10),'len rets 20',len(rets_20))
            return {
                'Ntraining': Ntraining_val,
                'Njump': Njump_val,
                'sigma_co': sigma_co_val,
                'sigma_ve': sigma_ve_val,
                'rets_05':rets_5,
                'rets_10':rets_10,
                'rets_20':rets_20,
                'len rets 5':len(rets_5),
                'len rets 10':len(rets_10),
                'len rets 20':len(rets_20),
                'Capital_top5': final_5,
                'Capital_top10': final_10,
                'Capital_top20': final_20,
                'Sharpe_top5': sharpe_5,
                'Sharpe_top10': sharpe_10,
                'Sharpe_top20': sharpe_20,
                'Median_top5': median_5,
                'Median_top10': median_10,
                'Median_top20': median_20,
                'Sharpe_top5Co': sharpe_5_co,
                'Sharpe_top10Co': sharpe_10_co,
                'Sharpe_top20Co': sharpe_20_co

            }
        except Exception as e:
            print(f"❌ Error en params {params}: {e}")
            return None

# Generar todas las combinaciones de hiperparámetros
param_combinations = list(product(Ntraining_l, Njump_l, sigma_co_l, sigma_ve_l))

# Ejecutar en paralelo
evaluator = EvaluateHyperParams(price, company, cnf)
with multiprocess.Pool(cnf.ncores) as pool:
    resultados = pool.map(evaluator, param_combinations)

# Filtrar resultados válidos
resultados = [r for r in resultados if r is not None]

# Convertir a DataFrame para análisis y guardado
df = pd.DataFrame(resultados)
csv_file = os.path.join(cnf.fname, 'Moving_Ntrain_18_07_hlp.csv')
df.to_csv(csv_file, index=False)
print(f"\n📄 Resultados guardados en '{csv_file}'")

# Opcional: gráficar resultados según quieras
# Ejemplo: capital final top 5 vs Ntraining, para un Njump, sigma_co y sigma_ve fijos (o alguna combinación)
