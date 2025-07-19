### Funcion que grafica los acumulados de capital en funcion del tiempo
import os
import numpy as np
import matplotlib.pyplot as plt



industrias = [
    "biotechnology", "machinery", "media", "metals", "oil", "construction",
    "defense", "insurance", "semiconductors", "software", "pharmaceuticals", "chemicals",
    "capital", "water", "beverages", "marine"
]
# Ruta del directorio con los archivos
for industria in industrias:
    base_dir = f"/home/ezequiel.geslin/datosmunin2/Trade/MyFinan2/ultimas_pruebas/test_industrias/prueba_{industria}/"

    # Iterar sobre todos los archivos del directorio
    for filename in os.listdir(base_dir):
        if filename.endswith("caps_array.npy"):
            # Ruta completa del archivo .npy
            file_path = os.path.join(base_dir, filename)

            # Cargar datos
            data = np.load(file_path)

            # Crear gráfico
            plt.figure(figsize=(12, 6))
            plt.plot(data[0], label='Top 5', color='blue')
            plt.plot(data[1], label='Top 10', color='green')
            plt.plot(data[2], label='Top 20', color='red')

            # Estética
            plt.title(f'Comparación: {filename}')
            plt.xlabel('Tiempo')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Guardar como PNG (mismo nombre base)
            output_name = filename.replace(".npy", ".png")
            output_path = os.path.join(base_dir, output_name)
            plt.savefig(output_path)
            plt.close()  # cerrar la figura para liberar memoria

            print(f"✅ Guardado: {output_path}")

    print("✅ Todos los gráficos generados.")
