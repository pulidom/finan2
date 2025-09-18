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
        Qz, Bz, By, centers_z = self.initialize_functional_spaces(z, y)
        
        for iter in range(n_iter):
            # Gradiente del coste de transporte (distancia cuadrática)
            cost_grad = y - x
            
            # Gradiente de penalización
            penalty_grad, s = self.compute_penalty_gradient(y, z, Qz, Bz, By)
            
            # Actualizar y
            y = y - 0.001 * (cost_grad + lambda_val * penalty_grad / n)
            
            if len(s) > 0 and np.max(s) < 0.1/np.sqrt(n):
                break
        
        return y, s
    
    def initialize_functional_spaces(self, z, y, n_centers=5):
        """Inicializa espacios funcionales F y G"""
        # Espacio F para z (parámetro)
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)
        
        n_centers = min(n_centers, len(z))
        kmeans_z = KMeans(n_clusters=n_centers, n_init=10, random_state=42).fit(z)
        centers_z = kmeans_z.cluster_centers_
        
        # Kernel gaussiano para F
        F = np.exp(-np.sum((z - centers_z.T)**2, axis=1) / (2 * 0.5**2))
        if len(F.shape) == 1:
            F = F.reshape(-1, 1)
        
        # Centrar F
        F_centered = F - np.mean(F, axis=0)
        
        # SVD para F
        if F_centered.shape[1] > 0:
            Uz, Sz, Vtz = svd(F_centered, full_matrices=False)
            Qz = Uz @ np.diag(Sz)
            Bz = Vtz.T / Sz
        else:
            Qz = np.zeros((len(z), 1))
            Bz = np.ones((1, 1))
        
        # Espacio G para y (observaciones) - funciones lineales y cuadráticas
        G = np.column_stack([np.ones_like(y), y, y**2])
        G_centered = G - np.mean(G, axis=0)
        
        # SVD para G
        Uy, Sy, Vty = svd(G_centered, full_matrices=False)
        Qy = Uy @ np.diag(Sy)
        By = Vty.T / Sy
        
        return Qz, Bz, By, centers_z
    
    def compute_penalty_gradient(self, y, z, Qz, Bz, By):
        """Calcula el gradiente del término de penalización"""
        try:
            # Recalcular G para los y actuales
            G_current = np.column_stack([np.ones_like(y), y, y**2])
            Qy_current = G_current @ By
            
            # Asegurar que las dimensiones coincidan
            if Qz.shape[0] != Qy_current.shape[0]:
                min_len = min(Qz.shape[0], Qy_current.shape[0])
                Qz = Qz[:min_len, :]
                Qy_current = Qy_current[:min_len, :]
            
            # Calcular matriz A y su SVD
            A = Qz.T @ Qy_current
            
            if A.size > 0:
                U, s, Vt = svd(A, full_matrices=False)
                
                # Calcular gradientes
                penalty_grad = np.zeros_like(y)
                for k in range(min(len(s), 3)):  # Limitar a máximo 3 componentes
                    a_k = U[:, k]
                    b_k = Vt[k, :]
                    
                    f_k = Qz @ a_k
                    dG = np.column_stack([np.zeros_like(y), np.ones_like(y), 2*y])
                    dg_k = dG @ (By @ b_k)
                    
                    penalty_grad += 2 * s[k] * f_k * dg_k
                
                return penalty_grad, s
            else:
                return np.zeros_like(y), np.array([0])
                
        except Exception as e:
            print(f"Error en compute_penalty_gradient: {e}")
            return np.zeros_like(y), np.array([0])

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
        try:
            mean_est, std_est = estimator.process_observation(obs, time_step, lambda_val=0.01, n_iter=30)
            estimated_means.append(mean_est)
            estimated_stds.append(std_est)
            
            if time_step % 2 == 0:
                print(f"Tiempo {time_step}: Media estimada = {mean_est:.3f}, Std estimada = {std_est:.3f}")
        except Exception as e:
            print(f"Error en tiempo {time_step}: {e}")
            # Usar valores del paso anterior si hay error
            if estimated_means:
                estimated_means.append(estimated_means[-1])
                estimated_stds.append(estimated_stds[-1])
            else:
                estimated_means.append(0)
                estimated_stds.append(1)

# Asegurar que tenemos suficientes estimaciones
while len(estimated_means) < time_steps:
    estimated_means.append(estimated_means[-1] if estimated_means else 0)
    estimated_stds.append(estimated_stds[-1] if estimated_stds else 1)

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

# Mostrar evolución de algunas estimaciones específicas
print("\nEvolución de estimaciones:")
for i in [0, 5, 10, 15, 19]:
    if i < len(estimated_means):
        print(f"Tiempo {i}: Media={estimated_means[i]:.3f} (verdadera={true_means[i]:.3f}), "
              f"Std={estimated_stds[i]:.3f} (verdadera={true_stds[i]:.3f})")