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

# Configuración base (se copiará por industria)
class cnf:
    pathdat = 'dat/'
    tipo = 'asset'
    mtd = 'on'
    beta_win = 121
    zscore_win = 41
    nmax = 20
    nsel = 20
    fname = '18_07/prueba_all'  # Se sobreescribirá por industria
    linver_betaweight = 0
    industry = 'marine'         # Se sobreescribirá por industria
    shorten = 0
    ncores = psutil.cpu_count(logical=False)

# Listas de hiperparámetros
Ntraining_l = [121, 131, 141, 151, 161]
Njump_l = [80]
sigma_co_l = [1.5]
sigma_ve_l = [0.1]
param_combinations = list(product(Ntraining_l, Njump_l, sigma_co_l, sigma_ve_l))

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
            caps_array = np.zeros_like(rets)
            caps_array[:, 0] = 100
            for i in range(3):
                caps_array[i, 1:] = caps_array[i, 0] * np.cumprod(1 + rets[i, 1:])

            # Guardar arrays numpy en el mismo directorio que el CSV
            base_dir = self.cnf.fname
            # Asegurarse que el directorio existe
            os.makedirs(base_dir, exist_ok=True)

            file_prefix = f"caps_array_Ntrain{Ntraining_val}_Njump{Njump_val}_sigmaCo{sigma_co_val}_sigmaVe{sigma_ve_val}"
            caps_array_path = os.path.join(base_dir, file_prefix + ".npy")
            rets_path = os.path.join(base_dir, file_prefix + "_rets.npy")

            np.save(caps_array_path, caps_array)
            np.save(rets_path, rets)

            final_5 = caps_array[0, -1]
            final_10 = caps_array[1, -1]
            final_20 = caps_array[2, -1]
            rets_5 = rets[0]
            rets_10 = rets[1]
            rets_20 = rets[2]
            sharpe_5 = np.mean(rets_5) / np.std(rets_5)
            sharpe_10 = np.mean(rets_10) / np.std(rets_10)
            sharpe_20 = np.mean(rets_20) / np.std(rets_20)
            median_5 = np.median(rets_5)
            median_10 = np.median(rets_10)
            median_20 = np.median(rets_20)
            sharpe_5_co = co.sharpe_ratio(caps_array[0])
            sharpe_10_co = co.sharpe_ratio(caps_array[1])
            sharpe_20_co = co.sharpe_ratio(caps_array[2])

            return {
                'Ntraining': Ntraining_val,
                'Njump': Njump_val,
                'sigma_co': sigma_co_val,
                'sigma_ve': sigma_ve_val,
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
                'Sharpe_top20Co': sharpe_20_co,
                'Rets_5': rets_5,
                'Rets_10': rets_10,
                'Rets_20': rets_20,
                'caps_array': caps_array  # Esto podés sacarlo si querés no guardar arrays grandes en el DataFrame
            }
        except Exception as e:
            print(f"❌ Error en params {params}: {e}")
            return None

# Lista de industrias a evaluar
industrias = [
    "biotechnology", "machinery", "media", "metals", "oil", "airlines", "construction",
    "defense", "insurance", "semiconductors", "software", "pharmaceuticals", "chemicals",
    "capital", "water", "beverages", "marine"
]

for industria in industrias:
    print(f"\n📊 Procesando industria: {industria}")

    class IndustriaConfig(copy.deepcopy(cnf)):
        industry = industria
        fname = f'18_07/test_industrias/prueba_{industria}'

    os.makedirs(IndustriaConfig.fname, exist_ok=True)

    day, date, price, company = load_ts(sector=IndustriaConfig.industry, pathdat=IndustriaConfig.pathdat)
    print(day.shape, date.shape, price.shape, company.shape)

    evaluator = EvaluateHyperParams(price, company, IndustriaConfig)
    with multiprocess.Pool(IndustriaConfig.ncores) as pool:
        resultados = pool.map(evaluator, param_combinations)

    resultados = [r for r in resultados if r is not None]

    df = pd.DataFrame(resultados)
    csv_file = os.path.join(IndustriaConfig.fname, f'Moving_Ntrain_{industria}.csv')
  #  df.to_csv(csv_file, index=False)
    print(f"✅ CSV guardado para {industria}: {csv_file}")

