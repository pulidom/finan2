import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from sklearn.cluster import KMeans
from scipy.linalg import svd

class LotkaVolterraEstimator:
    def __init__(self, prior_alpha=1.0, prior_beta=0.1, prior_gamma=1.5, prior_delta=0.075):
        """Inicializa el estimador para el modelo Lotka-Volterra"""
        self.prior_params = {
            'alpha': prior_alpha,
            'beta': prior_beta, 
            'gamma': prior_gamma,
            'delta': prior_delta
        }
        self.param_samples = None
        self.history = {'alpha': [], 'beta': [], 'gamma': [], 'delta': [], 'time': []}
    
    def initialize_parameter_samples(self, n_samples=500):
        """Inicializa muestras de parámetros desde distribuciones previas"""
        self.param_samples = {
            'alpha': np.random.normal(self.prior_params['alpha'], 0.1, n_samples),
            'beta': np.random.normal(self.prior_params['beta'], 0.01, n_samples),
            'gamma': np.random.normal(self.prior_params['gamma'], 0.1, n_samples),
            'delta': np.random.normal(self.prior_params['delta'], 0.01, n_samples)
        }
    
    def lotka_volterra(self, t, y, alpha, beta, gamma, delta):
        """Ecuaciones del modelo Lotka-Volterra"""
        x, y_val = y
        dxdt = alpha * x - beta * x * y_val
        dydt = delta * x * y_val - gamma * y_val
        return [dxdt, dydt]
    
    def simulate_observation(self, true_params, x0, y0, t_span, noise_std=0.1):
        """Simula una observación del sistema Lotka-Volterra"""
        sol = solve_ivp(
            self.lotka_volterra,
            t_span,
            [x0, y0],
            args=(true_params['alpha'], true_params['beta'], 
                  true_params['gamma'], true_params['delta']),
            dense_output=True
        )
        
        # Evaluar en el tiempo final con ruido
        t_final = t_span[1]
        x_final, y_final = sol.sol(t_final)
        
        # Añadir ruido observacional
        x_obs = x_final + np.random.normal(0, noise_std)
        y_obs = y_final + np.random.normal(0, noise_std)
        
        return np.array([x_obs, y_obs])
    
    def process_observation(self, observation, time_step, lambda_val=0.01, n_iter=20):
        """Procesa una nueva observación y actualiza la distribución de parámetros"""
        if self.param_samples is None:
            self.initialize_parameter_samples()
        
        # Crear datos para OTBP: z = parámetros, x = observación
        n_samples = len(self.param_samples['alpha'])
        z = np.column_stack([
            self.param_samples['alpha'],
            self.param_samples['beta'],
            self.param_samples['gamma'],
            self.param_samples['delta']
        ])
        
        x = np.tile(observation, (n_samples, 1))
        
        # Resolver problema de baricentro para cada dimensión de la observación
        y_transformed = np.zeros_like(x)
        for dim in range(x.shape[1]):
            y_dim, _ = self.solve_otbp(x[:, dim], z, lambda_val, n_iter)
            y_transformed[:, dim] = y_dim
        
        # Actualizar distribuciones de parámetros
        new_params = {}
        for i, param_name in enumerate(['alpha', 'beta', 'gamma', 'delta']):
            param_values = y_transformed[:, i] if i < y_transformed.shape[1] else z[:, i]
            new_params[param_name] = {
                'mean': np.mean(param_values),
                'std': np.std(param_values)
            }
        
        # Actualizar muestras para el siguiente paso
        for param_name in ['alpha', 'beta', 'gamma', 'delta']:
            self.param_samples[param_name] = np.random.normal(
                new_params[param_name]['mean'],
                new_params[param_name]['std'],
                n_samples
            )
        
        # Guardar historia
        for param_name in ['alpha', 'beta', 'gamma', 'delta']:
            self.history[param_name].append(new_params[param_name]['mean'])
        self.history['time'].append(time_step)
        
        return new_params
    
    def solve_otbp(self, x, z, lambda_val, n_iter):
        """Resuelve el problema de baricentro de transporte óptimo"""
        n = len(x)
        y = x.copy()
        
        # Inicializar espacios funcionales
        Qz, Bz, By, _ = self.initialize_functional_spaces(z, y)
        
        for iter in range(n_iter):
            # Gradiente del coste de transporte
            cost_grad = y - x
            
            # Gradiente de penalización
            penalty_grad, s = self.compute_penalty_gradient(y, z, Qz, Bz, By)
            
            # Actualizar y
            y = y - 0.001 * (cost_grad + lambda_val * penalty_grad / n)
            
            if len(s) > 0 and np.max(s) < 0.2/np.sqrt(n):
                break
        
        return y, s
    
    def initialize_functional_spaces(self, z, y, n_centers=3):
        """Inicializa espacios funcionales F y G"""
        # Espacio F para z (parámetros)
        n_centers = min(n_centers, len(z))
        kmeans_z = KMeans(n_clusters=n_centers, n_init=10, random_state=42).fit(z)
        centers_z = kmeans_z.cluster_centers_
        
        # Kernel gaussiano para F
        distances = np.linalg.norm(z[:, np.newaxis, :] - centers_z[np.newaxis, :, :], axis=2)
        F = np.exp(-distances**2 / (2 * 0.5**2))
        F_centered = F - np.mean(F, axis=0)
        
        # SVD para F
        if F_centered.shape[1] > 0:
            Uz, Sz, Vtz = svd(F_centered, full_matrices=False)
            Qz = Uz @ np.diag(Sz)
            Bz = Vtz.T / Sz
        else:
            Qz = np.ones((len(z), 1))
            Bz = np.ones((1, 1))
        
        # Espacio G para y (observaciones)
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
            
            # Asegurar dimensiones compatibles
            min_len = min(Qz.shape[0], Qy_current.shape[0])
            Qz = Qz[:min_len, :]
            Qy_current = Qy_current[:min_len, :]
            
            # Calcular matriz A
            A = Qz.T @ Qy_current
            
            if A.size > 0:
                U, s, Vt = svd(A, full_matrices=False)
                
                # Calcular gradientes
                penalty_grad = np.zeros_like(y)
                for k in range(min(len(s), 2)):  # Usar máximo 2 componentes
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

# Configuración del experimento Lotka-Volterra
np.random.seed(42)

# Parámetros verdaderos que varían con el tiempo (simulando cambios ambientales)
time_steps = 30
true_params_history = {
    'alpha': 0.8 + 0.4 * np.sin(np.linspace(0, 2*np.pi, time_steps)),  # Tasa de crecimiento presas
    'beta': 0.15 + 0.05 * np.cos(np.linspace(0, np.pi, time_steps)),   # Tasa de depredación
    'gamma': 1.2 + 0.6 * np.sin(np.linspace(0, 3*np.pi/2, time_steps)), # Tasa de mortalidad depredadores
    'delta': 0.06 + 0.04 * np.cos(np.linspace(0, np.pi, time_steps))   # Tasa de conversión
}

# Inicializar estimador
estimator = LotkaVolterraEstimator(
    prior_alpha=1.0,
    prior_beta=0.1,
    prior_gamma=1.5,
    prior_delta=0.075
)

# Condiciones iniciales
x0, y0 = 10.0, 5.0  # Presas y depredadores iniciales
t_span = (0, 5)      # Período de simulación entre observaciones

# Almacenar resultados
estimated_params = {'alpha': [], 'beta': [], 'gamma': [], 'delta': []}
observations_history = []

print("Simulando sistema Lotka-Volterra y estimando parámetros online...")

for time_step in range(time_steps):
    # Parámetros verdaderos en este paso de tiempo
    true_params = {key: true_params_history[key][time_step] for key in true_params_history}
    
    # Simular observación
    observation = estimator.simulate_observation(true_params, x0, y0, t_span, noise_std=0.2)
    observations_history.append(observation)
    
    # Actualizar condiciones iniciales para la próxima simulación
    x0, y0 = observation
    
    # Procesar observación y estimar parámetros
    try:
        new_params = estimator.process_observation(observation, time_step, lambda_val=0.005, n_iter=15)
        
        for param_name in ['alpha', 'beta', 'gamma', 'delta']:
            estimated_params[param_name].append(new_params[param_name]['mean'])
        
        if time_step % 5 == 0:
            print(f"Tiempo {time_step}:")
            for param_name in ['alpha', 'beta', 'gamma', 'delta']:
                print(f"  {param_name}: estimado={new_params[param_name]['mean']:.3f}, verdadero={true_params[param_name]:.3f}")
            print()
            
    except Exception as e:
        print(f"Error en tiempo {time_step}: {e}")
        # Usar valores anteriores si hay error
        for param_name in ['alpha', 'beta', 'gamma', 'delta']:
            if estimated_params[param_name]:
                estimated_params[param_name].append(estimated_params[param_name][-1])
            else:
                estimated_params[param_name].append(estimator.prior_params[param_name])

# Visualizar resultados
plt.figure(figsize=(16, 12))

# Parámetros individuales
param_names = ['alpha', 'beta', 'gamma', 'delta']
param_titles = ['Tasa de crecimiento presas (α)', 'Tasa de depredación (β)', 
                'Tasa de mortalidad depredadores (γ)', 'Tasa de conversión (δ)']

for i, param_name in enumerate(param_names):
    plt.subplot(3, 2, i+1)
    plt.plot(range(time_steps), estimated_params[param_name], 'b-', 
             label='Estimado', linewidth=2, marker='o', markersize=4)
    plt.plot(range(time_steps), true_params_history[param_name][:time_steps], 'r--', 
             label='Verdadero', linewidth=2)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor del parámetro')
    plt.title(param_titles[i])
    plt.legend()
    plt.grid(True, alpha=0.3)

# Evolución de las poblaciones
plt.subplot(3, 2, 5)
observations = np.array(observations_history)
plt.plot(range(time_steps), observations[:, 0], 'g-', label='Presas (estimadas)', linewidth=2)
plt.plot(range(time_steps), observations[:, 1], 'r-', label='Depredadores (estimados)', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Evolución de las poblaciones')
plt.legend()
plt.grid(True, alpha=0.3)

# Error de estimación
plt.subplot(3, 2, 6)
errors = []
for param_name in param_names:
    error = np.array(estimated_params[param_name]) - true_params_history[param_name][:time_steps]
    errors.append(np.sqrt(np.mean(error**2)))
    
plt.bar(range(len(param_names)), errors)
plt.xticks(range(len(param_names)), [p.upper() for p in param_names])
plt.xlabel('Parámetro')
plt.ylabel('Error RMS')
plt.title('Error de estimación por parámetro')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mostrar estadísticas finales
print("\n" + "="*60)
print("ESTADÍSTICAS FINALES DE ESTIMACIÓN - LOTKA-VOLTERRA")
print("="*60)

for param_name in param_names:
    true_vals = true_params_history[param_name][:time_steps]
    est_vals = estimated_params[param_name]
    error = np.array(est_vals) - true_vals
    
    print(f"\nParámetro {param_name.upper()}:")
    print(f"  Error RMS: {np.sqrt(np.mean(error**2)):.4f}")
    print(f"  Error absoluto medio: {np.mean(np.abs(error)):.4f}")
    print(f"  Error máximo: {np.max(np.abs(error)):.4f}")

# Simular el sistema con parámetros estimados finales para comparar
final_estimated_params = {key: estimated_params[key][-1] for key in param_names}

# Simular con parámetros verdaderos
sol_true = solve_ivp(
    estimator.lotka_volterra,
    (0, 20),
    [10, 5],
    args=(true_params_history['alpha'][-1], true_params_history['beta'][-1],
          true_params_history['gamma'][-1], true_params_history['delta'][-1]),
    dense_output=True
)

# Simular con parámetros estimados
sol_est = solve_ivp(
    estimator.lotka_volterra,
    (0, 20),
    [10, 5],
    args=(final_estimated_params['alpha'], final_estimated_params['beta'],
          final_estimated_params['gamma'], final_estimated_params['delta']),
    dense_output=True
)

# Visualizar comparación final
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
t_eval = np.linspace(0, 20, 100)
x_true, y_true = sol_true.sol(t_eval)
x_est, y_est = sol_est.sol(t_eval)

plt.plot(t_eval, x_true, 'g-', label='Presas (verdadero)', linewidth=2)
plt.plot(t_eval, y_true, 'r-', label='Depredadores (verdadero)', linewidth=2)
plt.plot(t_eval, x_est, 'g--', label='Presas (estimado)', linewidth=2)
plt.plot(t_eval, y_est, 'r--', label='Depredadores (estimado)', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Comparación: Comportamiento del sistema')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x_true, y_true, 'b-', label='Verdadero', linewidth=2)
plt.plot(x_est, y_est, 'r--', label='Estimado', linewidth=2)
plt.xlabel('Presas')
plt.ylabel('Depredadores')
plt.title('Espacio de fases')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()