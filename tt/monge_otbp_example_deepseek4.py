import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from sklearn.cluster import KMeans
from scipy.linalg import svd

class LotkaVolterraEstimator:
    def __init__(self, beta=0.1, delta=0.075, prior_alpha=1.0, prior_gamma=1.5):
        """Inicializa el estimador para el modelo Lotka-Volterra"""
        # Parámetros conocidos (fijos)
        self.beta = beta
        self.delta = delta
        
        # Parámetros a estimar
        self.prior_params = {
            'alpha': prior_alpha,
            'gamma': prior_gamma
        }
        self.param_samples = None
        self.history = {'alpha': [], 'gamma': [], 'time': []}
    
    def initialize_parameter_samples(self, n_samples=500):
        """Inicializa muestras de parámetros desde distribuciones previas"""
        self.param_samples = {
            'alpha': np.random.normal(self.prior_params['alpha'], 0.2, n_samples),
            'gamma': np.random.normal(self.prior_params['gamma'], 0.2, n_samples)
        }
    
    def lotka_volterra(self, t, y, alpha, gamma):
        """Ecuaciones del modelo Lotka-Volterra (β y δ fijos)"""
        x, y_val = y
        dxdt = alpha * x - self.beta * x * y_val
        dydt = self.delta * x * y_val - gamma * y_val
        return [dxdt, dydt]
    
    def simulate_observation(self, true_alpha, true_gamma, x0, y0, t_span, noise_std=0.1):
        """Simula una observación del sistema Lotka-Volterra"""
        sol = solve_ivp(
            self.lotka_volterra,
            t_span,
            [x0, y0],
            args=(true_alpha, true_gamma),
            dense_output=True,
            method='RK45'
        )
        
        # Evaluar en el tiempo final con ruido
        t_final = t_span[1]
        x_final, y_final = sol.sol(t_final)
        
        # Añadir ruido observacional
        x_obs = x_final + np.random.normal(0, noise_std)
        y_obs = y_final + np.random.normal(0, noise_std)
        
        return np.array([x_obs, y_obs])
    
    def process_observation(self, observation, time_step, lambda_val=0.05, n_iter=25):
        """Procesa una nueva observación y actualiza la distribución de parámetros"""
        if self.param_samples is None:
            self.initialize_parameter_samples()
        
        # Crear datos para OTBP: z = parámetros (alpha, gamma), x = observación
        n_samples = len(self.param_samples['alpha'])
        z = np.column_stack([
            self.param_samples['alpha'],
            self.param_samples['gamma']
        ])
        
        x = np.tile(observation, (n_samples, 1))
        
        # Resolver problema de baricentro para cada dimensión de la observación
        y_transformed = np.zeros_like(z)  # Transformamos los parámetros
        for dim in range(z.shape[1]):
            y_dim, _ = self.solve_otbp(z[:, dim], z, lambda_val, n_iter)
            y_transformed[:, dim] = y_dim
        
        # Actualizar distribuciones de parámetros
        new_params = {}
        param_names = ['alpha', 'gamma']
        for i, param_name in enumerate(param_names):
            new_params[param_name] = {
                'mean': np.mean(y_transformed[:, i]),
                'std': np.std(y_transformed[:, i])
            }
        
        # Actualizar muestras para el siguiente paso
        for param_name in param_names:
            self.param_samples[param_name] = np.random.normal(
                new_params[param_name]['mean'],
                max(new_params[param_name]['std'], 0.01),  # Evitar std demasiado pequeño
                n_samples
            )
        
        # Guardar historia
        for param_name in param_names:
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
            
            # Actualizar y con learning rate adaptativo
            lr = 0.001 * (0.9 ** iter)  # Learning rate decay
            y = y - lr * (cost_grad + lambda_val * penalty_grad / n)
            
            if len(s) > 0 and np.max(s) < 0.1/np.sqrt(n):
                break
        
        return y, s
    
    def initialize_functional_spaces(self, z, y, n_centers=4):
        """Inicializa espacios funcionales F y G"""
        # Espacio F para z (parámetros)
        n_centers = min(n_centers, len(z))
        if n_centers > 0:
            kmeans_z = KMeans(n_clusters=n_centers, n_init=10, random_state=42).fit(z)
            centers_z = kmeans_z.cluster_centers_
            
            # Kernel gaussiano para F
            distances = np.linalg.norm(z[:, np.newaxis, :] - centers_z[np.newaxis, :, :], axis=2)
            F = np.exp(-distances**2 / (2 * 0.3**2))
            F_centered = F - np.mean(F, axis=0)
            
            # SVD para F
            if F_centered.shape[1] > 0:
                Uz, Sz, Vtz = svd(F_centered, full_matrices=False)
                Qz = Uz @ np.diag(Sz)
                Bz = Vtz.T / Sz
            else:
                Qz = np.ones((len(z), 1))
                Bz = np.ones((1, 1))
        else:
            Qz = np.ones((len(z), 1))
            Bz = np.ones((1, 1))
            centers_z = np.array([[0, 0]])
        
        # Espacio G para y (parámetros transformados)
        G = np.column_stack([np.ones_like(y), y, y**2, np.sin(y), np.cos(y)])
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
            G_current = np.column_stack([np.ones_like(y), y, y**2, np.sin(y), np.cos(y)])
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
                for k in range(min(len(s), 3)):
                    a_k = U[:, k]
                    b_k = Vt[k, :]
                    
                    f_k = Qz @ a_k
                    # Derivadas de las funciones base
                    dG = np.column_stack([
                        np.zeros_like(y),
                        np.ones_like(y),
                        2*y,
                        np.cos(y),
                        -np.sin(y)
                    ])
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

# Parámetros conocidos (fijos)
beta_true = 0.1
delta_true = 0.075

# Parámetros verdaderos que varían con el tiempo
time_steps = 40
true_params_history = {
    'alpha': 0.8 + 0.4 * np.sin(np.linspace(0, 2*np.pi, time_steps)),  # Tasa de crecimiento presas
    'gamma': 1.2 + 0.6 * np.sin(np.linspace(0, 3*np.pi/2, time_steps)) # Tasa de mortalidad depredadores
}

# Inicializar estimador
estimator = LotkaVolterraEstimator(
    beta=beta_true,
    delta=delta_true,
    prior_alpha=1.0,
    prior_gamma=1.5
)

# Condiciones iniciales
x0, y0 = 20.0, 8.0  # Presas y depredadores iniciales
t_span = (0, 3)      # Período de simulación entre observaciones

# Almacenar resultados
estimated_params = {'alpha': [], 'gamma': []}
observations_history = []

print("Simulando sistema Lotka-Volterra y estimando parámetros α y γ online...")
print(f"Parámetros fijos: β = {beta_true}, δ = {delta_true}")
print()

for time_step in range(time_steps):
    # Parámetros verdaderos en este paso de tiempo
    true_alpha = true_params_history['alpha'][time_step]
    true_gamma = true_params_history['gamma'][time_step]
    
    # Simular observación
    observation = estimator.simulate_observation(
        true_alpha, true_gamma, x0, y0, t_span, noise_std=0.15
    )
    observations_history.append(observation)
    
    # Actualizar condiciones iniciales para la próxima simulación
    x0, y0 = observation
    
    # Procesar observación y estimar parámetros
    try:
        new_params = estimator.process_observation(observation, time_step, 
                                                  lambda_val=0.03, n_iter=20)
        
        estimated_params['alpha'].append(new_params['alpha']['mean'])
        estimated_params['gamma'].append(new_params['gamma']['mean'])
        
        if time_step % 5 == 0:
            print(f"Tiempo {time_step}:")
            print(f"  α: estimado={new_params['alpha']['mean']:.3f}, verdadero={true_alpha:.3f}")
            print(f"  γ: estimado={new_params['gamma']['mean']:.3f}, verdadero={true_gamma:.3f}")
            print()
            
    except Exception as e:
        print(f"Error en tiempo {time_step}: {e}")
        # Usar valores anteriores si hay error
        for param_name in ['alpha', 'gamma']:
            if estimated_params[param_name]:
                estimated_params[param_name].append(estimated_params[param_name][-1])
            else:
                estimated_params[param_name].append(estimator.prior_params[param_name])

# Visualizar resultados
plt.figure(figsize=(15, 10))

# Parámetros estimados vs verdaderos
plt.subplot(2, 2, 1)
plt.plot(range(time_steps), estimated_params['alpha'], 'b-', 
         label='α estimado', linewidth=2, marker='o', markersize=4)
plt.plot(range(time_steps), true_params_history['alpha'][:time_steps], 'b--', 
         label='α verdadero', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Valor del parámetro')
plt.title('Estimación del parámetro α (tasa de crecimiento presas)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(range(time_steps), estimated_params['gamma'], 'r-', 
         label='γ estimado', linewidth=2, marker='o', markersize=4)
plt.plot(range(time_steps), true_params_history['gamma'][:time_steps], 'r--', 
         label='γ verdadero', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Valor del parámetro')
plt.title('Estimación del parámetro γ (tasa de mortalidad depredadores)')
plt.legend()
plt.grid(True, alpha=0.3)

# Error de estimación
plt.subplot(2, 2, 3)
error_alpha = np.array(estimated_params['alpha']) - true_params_history['alpha'][:time_steps]
error_gamma = np.array(estimated_params['gamma']) - true_params_history['gamma'][:time_steps]

plt.plot(range(time_steps), error_alpha, 'b-', label='Error α', linewidth=2)
plt.plot(range(time_steps), error_gamma, 'r-', label='Error γ', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.title('Error de estimación')
plt.legend()
plt.grid(True, alpha=0.3)

# Evolución de las poblaciones
plt.subplot(2, 2, 4)
observations = np.array(observations_history)
plt.plot(range(time_steps), observations[:, 0], 'g-', label='Presas', linewidth=2)
plt.plot(range(time_steps), observations[:, 1], 'r-', label='Depredadores', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Evolución de las poblaciones observadas')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mostrar estadísticas finales
print("\n" + "="*60)
print("ESTADÍSTICAS FINALES DE ESTIMACIÓN - LOTKA-VOLTERRA")
print("="*60)

for param_name in ['alpha', 'gamma']:
    true_vals = true_params_history[param_name][:time_steps]
    est_vals = estimated_params[param_name]
    error = np.array(est_vals) - true_vals
    
    print(f"\nParámetro {param_name.upper()}:")
    print(f"  Error RMS: {np.sqrt(np.mean(error**2)):.4f}")
    print(f"  Error absoluto medio: {np.mean(np.abs(error)):.4f}")
    print(f"  Error máximo: {np.max(np.abs(error)):.4f}")
    print(f"  Correlación con verdadero: {np.corrcoef(est_vals, true_vals)[0,1]:.4f}")

# Comparación final del comportamiento del sistema
final_alpha_est = estimated_params['alpha'][-1]
final_gamma_est = estimated_params['gamma'][-1]
final_alpha_true = true_params_history['alpha'][-1]
final_gamma_true = true_params_history['gamma'][-1]

# Simular con parámetros verdaderos
sol_true = solve_ivp(
    estimator.lotka_volterra,
    (0, 25),
    [15, 6],
    args=(final_alpha_true, final_gamma_true),
    dense_output=True,
    method='RK45'
)

# Simular con parámetros estimados
sol_est = solve_ivp(
    estimator.lotka_volterra,
    (0, 25),
    [15, 6],
    args=(final_alpha_est, final_gamma_est),
    dense_output=True,
    method='RK45'
)

# Visualizar comparación final
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
t_eval = np.linspace(0, 25, 200)
x_true, y_true = sol_true.sol(t_eval)
x_est, y_est = sol_est.sol(t_eval)

plt.plot(t_eval, x_true, 'g-', label='Presas (verdadero)', linewidth=2)
plt.plot(t_eval, y_true, 'r-', label='Depredadores (verdadero)', linewidth=2)
plt.plot(t_eval, x_est, 'g--', label='Presas (estimado)', linewidth=2)
plt.plot(t_eval, y_est, 'r--', label='Depredadores (estimado)', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Comparación del comportamiento del sistema\n(parámetros finales)')
plt.legend()
plt.grid(True, alpha=0.3)

#plt.subplot(1, 2, 2)
#plt.plot(x

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
