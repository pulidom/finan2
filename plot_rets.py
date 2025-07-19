### Grafico de puntos de la ganancia/perdida por dia para los 5 10 y 20 mejores pares
import os
import numpy as np
import matplotlib.pyplot as plt

industrias = [
    "biotechnology", "machinery", "media", "metals", "oil", "construction",
    "defense", "insurance", "semiconductors", "software", "pharmaceuticals", "chemicals",
    "capital", "water", "beverages", "marine"
]

# Iterar por industria
for industria in industrias:
    base_dir = f"/home/ezequiel.geslin/datosmunin2/Trade/MyFinan2/ultimas_pruebas/test_industrias/prueba_{industria}/"

    for filename in os.listdir(base_dir):
        if filename.endswith("rets.npy"):
            file_path = os.path.join(base_dir, filename)
            data = np.load(file_path)

            # Crear figura con 3 subplots verticales
            fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

            colores = ['blue', 'green', 'red']
            titulos = ['Top 5', 'Top 10', 'Top 20']

            for i in range(3):
                axs[i].scatter(np.arange(data.shape[1]), data[i], color=colores[i], s=2)
                axs[i].set_ylabel('Valor')
                axs[i].set_title(titulos[i])
                axs[i].grid(True)

            axs[2].set_xlabel('Tiempo')
            fig.suptitle(f'Comparación: {filename}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # espacio para el título general

            output_name = filename.replace(".npy", "_scatter.png")
            output_path = os.path.join(base_dir, output_name)
            plt.savefig(output_path)
            plt.close()

            print(f"✅ Guardado: {output_path}")
    print(f"✅ Gráficos generados para industria: {industria}")
