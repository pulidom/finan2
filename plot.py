### Grafica los df de salida de las pruebas de hiperparametros en 3 paneles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
industrias = [
    "biotechnology", "machinery", "media", "metals", "oil", "airlines", "construction",
    "defense", "insurance", "semiconductors", "software", "pharmaceuticals", "chemicals",
    "capital", "water", "beverages", "marine"
]
for industria in industrias:
    df = pd.read_csv(f'/home/ezequiel.geslin/datosmunin2/Trade/MyFinan2/ultimas_pruebas/test_industrias/prueba_{industria}/Moving_Ntrain_{industria}.csv')
    print(np.unique(df['Ntraining']))
    print(df['Capital_top20'].max())
    print(df['Sharpe_top20'].max()*np.sqrt(252))

    fila_capital_max = df[df['Capital_top5'] == df['Capital_top5'].max()]
    print(fila_capital_max)

    fila_sharpe_max = df[df['Sharpe_top5'] == df['Sharpe_top5'].max()]
    print(fila_sharpe_max)

    output_dir = f"/home/ezequiel.geslin/datosmunin2/Trade/MyFinan2/ultimas_pruebas/test_industrias/prueba_{industria}/"
    os.makedirs(output_dir, exist_ok=True)

    fijos_personalizados = {
        'Ntraining': {'Njump': 80, 'sigma_co': 1.5, 'sigma_ve': 0.1},
        'Njump': {'Ntraining': 131, 'sigma_co': 1.5, 'sigma_ve': 0.1},
        'sigma_co': {'Ntraining': 131, 'Njump': 80, 'sigma_ve': 0.1},
        'sigma_ve': {'Ntraining': 131, 'Njump': 80, 'sigma_co': 1.5},
    }
    print("entrando a funcion")
    def graficar_3subplots(df, variable, fijos):
        condiciones = [(df[col] == val) for col, val in fijos.items()]
        filtro = condiciones[0]
        for cond in condiciones[1:]:
            filtro &= cond

        df_filtrado = df[filtro].sort_values(by=variable)

        if df_filtrado.empty:
            print(f"No hay datos para {variable} con los valores fijos especificados.")
            return

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # 1) Capital top5,10,20
        axs[0].plot(df_filtrado[variable], df_filtrado['Capital_top5'], label='Top 5', color='blue', marker='o')
        axs[0].plot(df_filtrado[variable], df_filtrado['Capital_top10'], label='Top 10', color='green', marker='s')
        axs[0].plot(df_filtrado[variable], df_filtrado['Capital_top20'], label='Top 20', color='red', marker='^')
        axs[0].set_ylabel('Capital')
        axs[0].set_title(f'Capital vs {variable}')
        axs[0].legend()
        axs[0].grid(True)
        # 2) Sharpe ratio anualizado (sharpe * sqrt(252))
        sqrt_252 = np.sqrt(252)
        axs[1].plot(df_filtrado[variable], df_filtrado['Sharpe_top5'] * sqrt_252, label='Top 5', color='blue', marker='o')
        axs[1].plot(df_filtrado[variable], df_filtrado['Sharpe_top10'] * sqrt_252, label='Top 10', color='green', marker='s')
        axs[1].plot(df_filtrado[variable], df_filtrado['Sharpe_top20'] * sqrt_252, label='Top 20', color='red', marker='^')
        axs[1].set_ylabel('Sharpe Anualizado')
        axs[1].set_title(f'Sharpe Ratio Anualizado vs {variable}')
        axs[1].legend()
        axs[1].grid(True)
        # 3) Medianas top5,10,20
        axs[2].plot(df_filtrado[variable], df_filtrado['Sharpe_top5Co'], label='Top 5', color='blue', marker='o')
        axs[2].plot(df_filtrado[variable], df_filtrado['Sharpe_top10Co'], label='Top 10', color='green', marker='s')
        axs[2].plot(df_filtrado[variable], df_filtrado['Sharpe_top20Co'], label='Top 20', color='red', marker='^')
        axs[2].set_ylabel('Sharpe Co')
        axs[2].set_xlabel(variable)
        axs[2].set_title(f'Sharpe Co vs {variable}')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        # Guardar gráfico
        filename = f"3subplots_vs_{variable}_18_07_hlp.png"
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        plt.close()
        print(f"Gráfico 3 subplots guardado en: {path}")
    # Ejemplo para 'Ntraining'
    #for variable, fijos in fijos_personalizados.items():
    #    graficar_3subplots(df, variable, fijos)
    graficar_3subplots(df, 'Ntraining', fijos_personalizados['Ntraining'])