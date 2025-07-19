### Para distintas configuraciones de hiperparametros ploteo el capital final
import numpy as np, os, copy
import matplotlib.pyplot as plt
from time import time
import pandas as pd

from read_data import load_ts
import arbitrage as ar
import cointegration as co

# Configuración
class cnf:
    pathdat = 'dat/'
    tipo = 'asset'  # 'asset', 'return', 'log_return', 'log'
    mtd = 'on'      # 'kf', 'exp', 'on', 'off'
    Njump = 86 # 86 es el mejor parece
#    Ntraining = 149
    beta_win = 121
    zscore_win = 41
    sigma_co = 1.5
    sigma_ve = 0.1
    nmax = -1
    nsel = 100
    fname = f"/home/ezequiel.geslin/datosmunin2/Trade/finan2/tmp2/"
    linver_betaweight = 0
    industry = 'beverages'
    shorten = 0

os.makedirs(cnf.fname, exist_ok=True)
# Valores de Njump para iterar
#Njump_values = list(range(21, 127, 5)) + list(range(136, 253, 10))
#Ntraining_values = list(range(121,150,2))+list(range(150,250,5))+list(range(250,500,10))+list(range(500,1000,30))
Ntraining_values = list(range(121,150,2))+list(range(150,200,5))
#Njump_values = list(range(50, 150, 5))
Ntraining_values = [121,126,131]
top_5_pval = []
top_10_pval = []
top_20_pval = []
# Cargar datos una vez
day, date, price, company = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)

# Guardar resultados
resultados = []

for Ntrain_val in Ntraining_values:
    print(f"\n⏳ Ejecutando con Ntrain = {Ntrain_val}")
    cnf.Ntraining = Ntrain_val

    caps = [[] for _ in range(3)]
    nt = price.shape[1]-756
    iini = 0

    # Iterar en ventanas móviles
    for ilast in range(cnf.Ntraining + cnf.Njump, nt, cnf.Njump):
        print(f" Ventana: {iini} ➝ {ilast} | Longitud: {ilast - iini}")
        assets_tr = price[:cnf.nmax, iini:ilast]
        iini += cnf.Njump

        # Calcular retornos de pares
        res = ar.all_pairs(assets_tr, company[:cnf.nmax], cnf)
#        print(dir(res))
        # Seleccionar mejores por p-value
        metrics = co.all_pairs_stats(assets_tr[:, :ilast - cnf.Njump], company, 'asset')
        idx = np.argsort(metrics.half_life)[:cnf.nsel] 
        res.reorder(idx)

        # Guardar retornos promedio para top 5, 10, 20 pares
        caps[0].append(res.retorno[:5, cnf.Ntraining:].mean(0))
        caps[1].append(res.retorno[:10, cnf.Ntraining:].mean(0))
        caps[2].append(res.retorno[:20, cnf.Ntraining:].mean(0))

    # Convertir retornos a arrays
    rets = np.array([np.concatenate(cap) for cap in caps])
    caps_array = np.zeros_like(rets)
    caps_array[:, 0] = 100
    for i in range(3):
        caps_array[i, 1:] = caps_array[i, 0] * np.cumprod(1 + rets[i, 1:])

    # Extraer capital final
    final_5 = caps_array[0, -1]
    final_10 = caps_array[1, -1]
    final_20 = caps_array[2, -1]
    top_5_pval.append(final_5)
    top_10_pval.append(final_10)
    top_20_pval.append(final_20)    
    # Imprimir
    print(f"✅ Capital final top 5 pval: {final_5:.2f}")
    print(f"✅ Capital final top 10 pval: {final_10:.2f}")
    print(f"✅ Capital final top 20 pval: {final_20:.2f}")

    # Guardar resultados en lista
    resultados.append({
        'Njump': Ntrain_val,
        'Capital_top5': final_5,
        'Capital_top10': final_10,
        'Capital_top20': final_20
    })
print("eje x: ",Ntraining_values)
print("ultimos valores de 5: ",top_5_pval)
print("ultimos valores de 10: ",top_10_pval)
print("ultimos valores de 20: ",top_20_pval)


plt.figure(figsize=(8, 5))
plt.plot(Ntraining_values, top_5_pval, marker='o', color='blue', label='Top 5 pval')
plt.plot(Ntraining_values, top_10_pval, marker='s', color='orange', label='Top 10 pval')
plt.plot(Ntraining_values, top_20_pval, marker='^', color='green', label='Top 20 pval')


# Títulos y ejes
plt.title("Seleccion por HALF_LIFE PENALTY Njump = 86")
plt.xlabel("Njump (experimentos)")
plt.ylabel("Capital final")
plt.grid(True)
plt.legend()

# Guardar
file_plot = "/home/ezequiel.geslin/datosmunin2/Trade/finan2/tmp2/"
plt.tight_layout()
plt.savefig(file_plot + "half_life_testing_penalty.png")
plt.close()

print(f"✅ Gráfico guardado como {file_plot}.png")

# Guardar en CSV
#df = pd.DataFrame(resultados)
#csv_file = cnf.fname+'resultados4_pval.csv'
#df.to_csv(csv_file, index=False)
#print(f"\n📄 Resultados guardados en '{csv_file}'")


