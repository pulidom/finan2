import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
from scipy.linalg import svd

class OnlineParameterEstimator:
    def __init__(self, prior_mean=0, prior_std=1):
        """Inicializa el estimador con una distribución previa"""
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.parameter_samples = None
        self.history = {'mean': [], 'std': [], 'time': []}
    
    def initialize_parameter_samples(self, n_samples=1000):
        """Inicializa muestras del parámetro desde la distribución previa"""
        self.parameter_samples = np.random.normal(
            self.prior_mean, self.prior_std, n_samples
        )
    
    def process_observation(self, observation, time_step, lambda_val=0.1, n_iter=50):
        """Procesa una nueva observación y actualiza la distribución del parámetro"""
        if self.parameter_samples is None:
            self.initialize_parameter_samples()
        
        # Paso 1: Crear datos para el problema OTBP
        # z = parámetro (alpha), x = observación
        z = self.parameter_samples.reshape(-1, 1)
        x = np.array([observation] * len(z))
        
        # Resolver problema de baricentro
        y, _ = self.solve_otbp(x, z, lambda_val, n_iter)
        
        # Paso 2: Actualizar la distribución usando Bayes (a través del baricentro)
        # La nueva distribución es proporcional a p(x|z) * p(z)
        # Aproximamos esto ajustando una distribución normal a las muestras transformadas
        new_mean = np.mean(y)
        new_std = np.std(y)
        
        # Actualizar muestras para el siguiente paso temporal
        self.parameter_samples = np.random.normal(new_mean, new_std, len(self.parameter_samples))
        
        # Guardar historia
        self.history['mean'].append(new_mean)
        self.history['std'].append(new_std)
        self.history['time'].append(time_step)
        
        return new_mean, new_std
    
    def solve_otbp(self, x, z, lambda_val, n_iter):
        """Resuelve el problema de baricentro de transporte óptimo"""
        n = len(x)
        y = x.copy()
        
        # Inicializar espacios funcionales
        Qz, Bz, Qy_init, By, _ = self.initialize_functional_spaces(z, y)
        
        for iter in range(n_iter):
            # Gradiente del coste de transporte (distancia cuadrática)
            cost_grad = y - x
            
            # Gradiente de penalización
            penalty_grad, s = self.compute_penalty_gradient(y, z, Qz, Bz, By)
            
            # Actualizar y
            y = y - 0.001 * (cost_grad + lambda_val * penalty_grad / n)
            
            if np.max(s) < 0.1/np.sqrt(n):
                break
        
        return y, s
    
    def initialize_functional_spaces(self, z, y, n_centers=5):
        """Inicializa espacios funcionales F y G"""
        # Espacio F para z (parámetro)
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)
        
        kmeans_z = KMeans(n_clusters=min(n_centers, len(z)), n_init=10).fit(z)
        centers_z = kmeans_z.cluster_centers_
        
        # Kernel gaussiano para F
        F = np.exp(-np.sum((z - centers_z.T)**2, axis=1) / (2 * 0.5**2))
        F = F.reshape(-1, centers_z.shape[0])
        
        # Centrar y ortogonalizar F
        F_centered = F - np.mean(F, axis=0)
        Uz, Sz, Vtz = svd(F_centered, full_matrices=False)
        Qz = Uz @ np.diag(Sz)
        Bz = Vtz.T / Sz
        
        # Espacio G para y (observaciones) - funciones lineales y cuadráticas
        G = np.column_stack([y, y**2])
        G_centered = G - np.mean(G, axis=0)
        Uy, Sy, Vty = svd(G_centered, full_matrices=False)
        Qy = Uy @ np.diag(Sy)
        By = Vty.T / Sy
        
        return Qz, Bz, Qy, By, centers_z
    
    def compute_penalty_gradient(self, y, z, Qz, Bz, By):
        """Calcula el gradiente del término de penalización"""
        # Recalcular G para los y actuales
        G_current = np.column_stack([y, y**2])
        Qy_current = G_current @ By
        
        # Calcular matriz A y su SVD
        A = Qz.T @ Qy_current
        U, s, Vt = svd(A, full_matrices=False)
        
        # Calcular gradientes
        penalty_grad = np.zeros_like(y)
        for k in range(len(s)):
            a_k = U[:, k]
            b_k = Vt[k, :]
            
            f_k = Qz @ a_k
            dG = np.column_stack([np.ones_like(y), 2*y])
            dg_k = dG @ (By @ b_k)
            
            penalty_grad += 2 * s[k] * f_k * dg_k
        
        return penalty_grad, s

# Función para generar datos de un proceso con parámetro variable
def generate_time_series_data(true_means, true_stds, n_observations=100):
    """Genera una serie temporal con parámetros que varían en el tiempo"""
    observations = []
    true_params = []
    
    for i in range(len(true_means)):
        # Generar múltiples observaciones para cada paso de tiempo
        obs_at_time = np.random.normal(
            true_means[i], true_stds[i], n_observations
        )
        observations.extend(obs_at_time)
        true_params.extend([(true_means[i], true_stds[i])] * n_observations)
    
    return np.array(observations), np.array(true_params)

# Configuración del experimento
np.random.seed(42)

# Parámetros verdaderos que varían con el tiempo
time_steps = 20
true_means = np.linspace(0, 5, time_steps)  # Media aumenta linealmente
true_stds = 0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, time_steps))  # Desviación oscila

# Generar datos
observations, true_params = generate_time_series_data(true_means, true_stds, n_observations=10)

# Inicializar estimador
estimator = OnlineParameterEstimator(prior_mean=0, prior_std=1)

# Procesar observaciones secuencialmente
estimated_means = []
estimated_stds = []

print("Procesando observaciones secuencialmente...")
for i, obs in enumerate(observations):
    if i % 10 == 0:  # Actualizar cada 10 observaciones (cada paso de tiempo)
        time_step = i // 10
        mean_est, std_est = estimator.process_observation(obs, time_step)
        estimated_means.append(mean_est)
        estimated_stds.append(std_est)
        
        if time_step % 5 == 0:
            print(f"Tiempo {time_step}: Media estimada = {mean_est:.3f}, Std estimada = {std_est:.3f}")

# Visualizar resultados
plt.figure(figsize=(15, 10))

# Media vs tiempo
plt.subplot(2, 2, 1)
plt.plot(range(len(estimated_means)), estimated_means, 'b-', label='Media estimada', linewidth=2)
plt.plot(range(len(true_means)), true_means, 'r--', label='Media verdadera', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Media')
plt.title('Evolución de la media estimada')
plt.legend()
plt.grid(True, alpha=0.3)

# Desviación estándar vs tiempo
plt.subplot(2, 2, 2)
plt.plot(range(len(estimated_stds)), estimated_stds, 'g-', label='Std estimada', linewidth=2)
plt.plot(range(len(true_stds)), true_stds, 'r--', label='Std verdadera', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Desviación estándar')
plt.title('Evolución de la desviación estándar estimada')
plt.legend()
plt.grid(True, alpha=0.3)

# Error de estimación de la media
plt.subplot(2, 2, 3)
error_means = np.array(estimated_means) - true_means[:len(estimated_means)]
plt.plot(range(len(error_means)), error_means, 'orange', label='Error de media', linewidth=2)
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.title('Error en la estimación de la media')
plt.legend()
plt.grid(True, alpha=0.3)

# Error de estimación de la desviación estándar
plt.subplot(2, 2, 4)
error_stds = np.array(estimated_stds) - true_stds[:len(estimated_stds)]
plt.plot(range(len(error_stds)), error_stds, 'purple', label='Error de std', linewidth=2)
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.title('Error en la estimación de la desviación estándar')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mostrar estadísticas finales
print("\n" + "="*50)
print("ESTADÍSTICAS FINALES DE ESTIMACIÓN")
print("="*50)
print(f"Error RMS media: {np.sqrt(np.mean(error_means**2)):.4f}")
print(f"Error RMS desviación estándar: {np.sqrt(np.mean(error_stds**2)):.4f}")
print(f"Error absoluto medio (media): {np.mean(np.abs(error_means)):.4f}")
print(f"Error absoluto medio (std): {np.mean(np.abs(error_stds)):.4f}")