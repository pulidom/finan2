## Ejecutar ejemplo
#if __name__ == "__main__":
#    # Generar datos
#    print("Generando datos del Ejemplo 9.1...")
#    x, z = generate_example_data(1500)
#    
#    # Crear y ajustar modelo con monitoreo en tiempo real
#    print("\nAjustando modelo OTBP con monitoreo en tiempo real...")
#    print("Los gráficos se actualizarán cada 50 iteraciones.")
#    print("Puedes ver la convergencia en tiempo real.")
#    
#    model = MongeOTBP(K_max=3)
#    
#    # Ajustar con visualización interactiva
#    model.fit(x, z, max_iterations=500, verbose=True, 
#              plot_every=20, interactive=True)  # Actualizar cada 20 iteraciones
#    
#    # Generar gráficos finales de resultados
#    print("\nGenerando gráficos de resultados finales...")
#    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#    
#    # Datos originales
#    axes[0, 0].scatter(z[:, 0], x, alpha=0.5, s=1)
#    axes[0, 0].set_title('Datos Originales x vs z1')
#    axes[0, 0].set_xlabel('z1')
#    axes[0, 0].set_ylabel('x')
#    
#    # Barycenter
#    axes[1, 0].hist(model.y_final, bins=50, alpha=0.7, density=True)
#    axes[1, 0].set_title('Barycenter y')
#    axes[1, 0].set_xlabel('y')
#    axes[1, 0].set_ylabel('Densidad')
#    
#    # Simulación condicional para z* = (0.2, 0.3)
#    z_star1 = np.array([0.2, 0.3])
#    x_sim1 = model.simulate_conditional(z_star1, n_samples=500)
#    
#    axes[0, 1].hist(x_sim1, bins=30, alpha=0.7, density=True, label='Simulado')
#    
#    # Densidad verdadera para comparación
#    mu_true1 = np.cos(2*np.pi*0.2) + np.sin(np.pi*0.3)
#    sigma_true1 = 0.2 * np.sqrt((1-2*0.2) * (1-2*0.3))
#    x_range1 = np.linspace(x_sim1.min(), x_sim1.max(), 100)
#    true_density1 = (1/np.sqrt(2*np.pi*sigma_true1**2)) * np.exp(-0.5*((x_range1-mu_true1)/sigma_true1)**2)
#    
#    axes[0, 1].plot(x_range1, true_density1, 'r-', label='Verdadero', linewidth=2)
#    axes[0, 1].set_title('ρ(x|z*) para z*=(0.2, 0.3)')
#    axes[0, 1].set_xlabel('x')
#    axes[0, 1].set_ylabel('Densidad')
#    axes[0, 1].legend()
#    
#    # Simulación condicional para z* = (-0.2, -0.3)
#    z_star2 = np.array([-0.2, -0.3])
#    x_sim2 = model.simulate_conditional(z_star2, n_samples=500)
#    
#    axes[1, 1].hist(x_sim2, bins=30, alpha=0.7, density=True, label='Simulado')
#    
#    # Densidad verdadera para comparación
#    mu_true2 = np.cos(2*np.pi*(-0.2)) + np.sin(np.pi*(-0.3))
#    sigma_true2 = 0.2 * np.sqrt((1-2*(-0.2)) * (1-2*(-0.3)))
#    x_range2 = np.linspace(x_sim2.min(), x_sim2.max(), 100)
#    true_density2 = (1/np.sqrt(2*np.pi*sigma_true2**2)) * np.exp(-0.5*((x_range2-mu_true2)/sigma_true2)**2)
#    
#    axes[1, 1].plot(x_range2, true_density2, 'r-', label='Verdadero', linewidth=2)
#    axes[1, 1].set_title('ρ(x|z*) para z*=(-0.2, -0.3)')
#    axes[1, 1].set_xlabel('x')
#    axes[1, 1].set_ylabel('Densidad')
#    axes[1, 1].legend()
#    
#    # Gráficos de convergencia basados en datos guardados
#    conv_data = model.convergence_data
#    iterations = conv_data['iterations']
#    
#    axes[0, 2].plot(iterations, conv_data['total_loss'])
#    axes[0, 2].set_title('Convergencia: Pérdida Total')
#    axes[0, 2].set_xlabel('Iteración')
#    axes[0, 2].set_ylabel('Pérdida')
#    axes[0, 2].set_yscale('log')
#    axes[0, 2].grid(True, alpha=0.3)
#    
#    axes[1, 2].plot(iterations, conv_data['max_sigma'])
#    axes[1, 2].axhline(y=model.nu/np.sqrt(len(x)), color='r', linestyle='--', 
#                      label=f'Umbral σ* = {model.nu/np.sqrt(len(x)):.4f}')
#    axes[1, 2].set_title('Convergencia: Valor Singular Máximo')
#    axes[1, 2].set_xlabel('Iteración')
#    axes[1, 2].set_ylabel('σ_max')
#    axes[1, 2].legend()
#    axes[1, 2].set_yscale('log')
#    axes[1, 2].grid(True, alpha=0.3)
#    
#    plt.tight_layout()
#    plt.show()
#    
#    # Imprimir estadísticas finales
#    print(f"\nResultados finales:")
#    print(f"Valor singular máximo: {conv_data['max_sigma'][-1]:.6f}")
#    print(f"Umbral σ*: {model.nu/np.sqrt(len(x)):.6f}")
#    print(f"Lambda final: {conv_data['lambda_vals'][-1]:.3f}")
#    print(f"Número total de iteraciones: {len(conv_data['iterations'])}")
#    
#    # Función para ejecutar con diferentes parámetros
#    def run_quick_test():
#        """Ejecutar una prueba rápida sin gráficos interactivos"""
#        print("\n" + "="*50)
#        print("EJECUTANDO PRUEBA RÁPIDA (sin visualización)")
#        print("="*50)
#        
#        x_test, z_test = generate_example_data(300)  # Menos datos para ser más rápido
#        model_test = MongeOTBP(K_max=2)
#        
#        start_time = time.time()
#        model_test.fit(x_test, z_test, max_iterations=200, verbose=True, 
#                      interactive=False)  # Sin gráficos interactivos
#        end_time = time.time()
#        
#        print(f"Tiempo total: {end_time - start_time:.2f} segundos")
#        print(f"Convergencia: {'SÍ' if len(model_test.convergence_data['max_sigma']) > 0 and model_test.convergence_data['max_sigma'][-1] < model_test.nu/np.sqrt(len(x_test)) else 'NO'}")
#        
#        return model_test
#    
#    # Preguntar al usuario si quiere ejecutar la prueba rápida
#    print("\n" + "="*60)
#    print("¿Te gustaría ejecutar una prueba rápida sin visualización?")
#    print("Esto te permitirá ver qué tan rápido converge el algoritmo.")
#    print("="*60)
#    
#    # Ejecutar automáticamente la prueba rápida para demostración
#    quick_model = run_quick_test()

import numpy as np
    
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MongeOTBP:
    """
    Implementación del Monge Optimal Transport Barycenter Problem
    basado en el paper de Lipnick et al.
    """
    
    def __init__(self, lambda_param=None, alpha=0.0025, delta=0.1, 
                 nu=0.2, theta=1.1, kappa=0.5, tau=0.5, K_max=5):
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.delta = delta
        self.nu = nu
        self.theta = theta
        self.kappa = kappa
        self.tau = tau
        self.K_max = K_max
        self.history = []
    
    def _gaussian_kernel(self, x, centers, bandwidth):
        """Función kernel gaussiana"""
        x = np.atleast_2d(x)
        centers = np.atleast_2d(centers)
        
        if x.shape[1] != centers.shape[1]:
            x = x.T if x.shape[0] == centers.shape[1] else x
            
        dist_sq = np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-dist_sq / (2 * bandwidth ** 2))
    
    def _create_function_spaces(self, data, n_centers=20, bandwidth=None):
        """Crear espacios funcionales F y G usando kernels gaussianos"""
        n_samples = len(data)
        
        if bandwidth is None:
            bandwidth = np.std(data) / 2
        
        # Usar k-means para encontrar centros
        if len(data.shape) == 1:
            data_reshaped = data.reshape(-1, 1)
        else:
            data_reshaped = data
            
        n_centers = min(n_centers, n_samples)
        
        if n_centers == 1:
            centers = np.mean(data_reshaped, axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
            centers = kmeans.fit(data_reshaped).cluster_centers_
        
        # Crear matriz de funciones
        func_matrix = self._gaussian_kernel(data_reshaped, centers, bandwidth)

        #func_matrix = func_matrix - np.mean(func_matrix, axis=0)# centro deepseek
 
        return func_matrix, centers, bandwidth
    
    def _orthogonalize_functions(self, func_matrix):
        """Ortogonalizar matriz de funciones usando SVD"""
        U, s, Vt = svd(func_matrix, full_matrices=False)
        
        # Determinar rango efectivo
        s_squared = s ** 2
        total_energy = np.sum(s_squared)
        cumsum_energy = np.cumsum(s_squared)
        
        # Mantener componentes que explican al menos 99% de la energía
        threshold = 0.99 * total_energy
        effective_rank = np.searchsorted(cumsum_energy, threshold) + 1
        effective_rank = min(effective_rank, len(s))
        
        # Crear matriz ortogonal
        Q = U[:, :effective_rank] #@ np.diag(Sz)????
        B = (Vt[:effective_rank, :] / s[:effective_rank, np.newaxis]).T
        
        return Q, B
    
    def _compute_penalty_matrix(self, Qz, Qy):
        """Calcular matriz A para el término de penalización"""
        return Qz.T @ Qy
    
    def _compute_singular_values_and_vectors(self, A):
        """Calcular valores y vectores singulares de A"""
        U, s, Vt = svd(A, full_matrices=False)
        K = min(len(s), self.K_max)
        return s[:K], U[:, :K], Vt[:K, :].T
    
    def _update_lambda(self, grad_cost_norm, grad_penalty_norm, var_x):
        """Actualizar parámetro de penalización λ"""
        numerator = np.sqrt(grad_cost_norm + 0.1 * var_x)
        denominator = np.sqrt(grad_penalty_norm + 1e-10)
        return 0.5 * numerator / denominator
    
    def _line_search(self, y, grad, L_current):
        """Búsqueda de línea usando algoritmo de Armijo-Goldstein"""
        eta = 0.1  # learning rate inicial
        
        for _ in range(10):
            y_new = y - eta * grad
            L_new = self._compute_objective(y_new)
            
            # Condición de Armijo
            if L_new <= L_current - self.kappa * eta * np.sum(grad ** 2):
                return eta, y_new
            
            eta *= self.tau
        
        return eta, y - eta * grad
    
    def _compute_cost_gradient(self, x, y):
        """Calcular gradiente del costo (para costo cuadrático)"""
        return y - x
    
    def _compute_penalty_gradient(self, Qy, By, sigma_vals, a_vecs, b_vecs):
        """Calcular gradiente del término de penalización"""
        n_samples = Qy.shape[0]
        grad = np.zeros(n_samples)
        
        for k in range(len(sigma_vals)):
            # Gradiente de cada valor singular
            sigma_grad = np.zeros(n_samples)
            for i in range(n_samples):
                for j in range(By.shape[1]):
                    sigma_grad[i] += a_vecs[k, j] * By[j, :] @ b_vecs[:, k]
            
            grad += 2 * sigma_vals[k] * sigma_grad
        
        return grad
    
    def _compute_objective(self, y):
        """Calcular función objetivo completa"""
        # Recalcular matrices y valores singulares para y actual
        Fy_current, _, _ = self._create_function_spaces(y, n_centers=min(20, len(y)))
        Qy_current, _ = self._orthogonalize_functions(Fy_current)
        
        A_current = self._compute_penalty_matrix(self.Qz, Qy_current)
        sigma_vals, _, _ = self._compute_singular_values_and_vectors(A_current)
        
        cost_term = 0.5 * np.sum((self.x - y) ** 2)
        penalty_term = self.lambda_current * np.sum(sigma_vals ** 2)
        
        return cost_term + penalty_term
    
    def fit(self, x, z, max_iterations=1000, verbose=True):
        """
        Ajustar el modelo OTBP
        
        Parameters:
        -----------
        x : array-like, shape (n_samples, n_features)
            Variables de resultado
        z : array-like, shape (n_samples, n_factors) 
            Variables factoriales/covariables
        max_iterations : int
            Número máximo de iteraciones
        verbose : bool
            Mostrar progreso
        """
        self.x = np.array(x).reshape(-1) if len(np.array(x).shape) == 1 else np.array(x)
        self.z = np.array(z)
        
        if len(self.x.shape) == 1:
            self.x = self.x.reshape(-1, 1)
        if len(self.z.shape) == 1:
            self.z = self.z.reshape(-1, 1)
        
        n_samples = len(self.x)
        
        # Inicializar y = x
        y = self.x.copy()
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        
        # Crear espacios funcionales para z (fijos durante el algoritmo)
        Fz, self.z_centers, self.z_bandwidth = self._create_function_spaces(self.z)
        
        # Asegurar que las funciones tengan media cero
        Fz_mean = np.mean(Fz, axis=0)
        Fz = Fz - Fz_mean
        
        self.Qz, self.Bz = self._orthogonalize_functions(Fz)
        
        var_x = np.var(self.x)
        self.lambda_current = 1.0
        
        y0 = y.copy()  # Para criterio de nueva etapa
        stage_count = 0
        
        for iteration in range(max_iterations):
            # Crear espacios funcionales para y (se actualizan)
            if iteration == 0 or iteration % 50 == 0:  # Actualizar cada 50 iteraciones
                # Para el ejemplo 9.1, usamos funciones lineales y cuadráticas
                Fy = np.column_stack([y, y**2])  # Espacios funcionales simples
                self.Qy, self.By = self._orthogonalize_functions(Fy)
            
            # Calcular matriz de penalización
            A = self._compute_penalty_matrix(self.Qz, self.Qy)
            
            # Obtener valores y vectores singulares
            sigma_vals, a_vecs, b_vecs = self._compute_singular_values_and_vectors(A)
            
            # Calcular gradientes
            grad_cost = self._compute_cost_gradient(self.x.reshape(-1), y)
            
            # Gradiente de penalización simplificado
            grad_penalty = np.zeros_like(y)
            for k in range(len(sigma_vals)):
                # Aproximación del gradiente usando diferencias finitas
                eps = 1e-8
                y_plus = y.copy()
                y_plus += eps
                
                Fy_plus = np.column_stack([y_plus, y_plus**2])
                Qy_plus, _ = self._orthogonalize_functions(Fy_plus)
                A_plus = self._compute_penalty_matrix(self.Qz, Qy_plus)
                s_plus, _, _ = self._compute_singular_values_and_vectors(A_plus)
                
                if k < len(s_plus):
                    grad_penalty += 2 * sigma_vals[k] * (s_plus[k] - sigma_vals[k]) / eps
            
            # Actualizar lambda
            grad_cost_norm = np.linalg.norm(grad_cost) ** 2
            grad_penalty_norm = np.linalg.norm(grad_penalty) ** 2
            
            if grad_penalty_norm > 1e-10:
                self.lambda_current = self._update_lambda(grad_cost_norm, grad_penalty_norm, var_x)
            
            # Gradiente total
            grad_total = grad_cost + self.lambda_current * grad_penalty
            
            # Búsqueda de línea
            L_current = self._compute_objective(y)
            eta, y_new = self._line_search(y, grad_total, L_current)
            y = y_new
            
            # Guardar historial
            cost_term = 0.5 * np.sum((self.x.reshape(-1) - y) ** 2)
            penalty_term = self.lambda_current * np.sum(sigma_vals ** 2)
            total_loss = cost_term + penalty_term
            
            self.history.append({
                'iteration': iteration,
                'cost': cost_term,
                'penalty': penalty_term,
                'total_loss': total_loss,
                'lambda': self.lambda_current,
                'max_sigma': np.max(sigma_vals) if len(sigma_vals) > 0 else 0
            })
            
            # Criterios de terminación
            sigma_threshold = self.nu / np.sqrt(n_samples)
            grad_threshold = self.alpha * var_x
            
            max_sigma = np.max(sigma_vals) if len(sigma_vals) > 0 else 0
            grad_norm = np.sum(grad_total ** 2) / n_samples
            
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Loss={total_loss:.6f}, Max σ={max_sigma:.6f}, "
                      f"Grad norm={grad_norm:.6f}, λ={self.lambda_current:.3f}")
            
            # Verificar convergencia
            if max_sigma < sigma_threshold and grad_norm < grad_threshold:
                if verbose:
                    print(f"Convergencia alcanzada en iteración {iteration}")
                break
            
            # Criterio para nueva etapa
            stage_diff = np.sum((y - y0) ** 2)
            stage_var = np.sum((y0 - np.mean(y0)) ** 2)
            
            if stage_diff > self.delta * stage_var:
                y0 = y.copy()
                stage_count += 1
                if verbose:
                    print(f"Nueva etapa {stage_count} en iteración {iteration}")
        
        self.y_final = y
        self.final_sigma_vals = sigma_vals
        self.final_a_vecs = a_vecs
        self.final_b_vecs = b_vecs
        
        return self
    
    def simulate_conditional(self, z_target, n_samples=None):
        """
        Simular muestras de ρ(x|z*) para un valor objetivo z*
        """
        if n_samples is None:
            n_samples = len(self.y_final)
        
        # Usar las muestras del barycenter
        y_samples = self.y_final[:n_samples]
        
        # Fórmula de inversión (5.1) simplificada
        z_target = np.array(z_target)
        if len(z_target.shape) == 0:
            z_target = z_target.reshape(1)
        
        # Evaluar funciones en z_target
        z_target_reshaped = z_target.reshape(1, -1)
        Fz_target = self._gaussian_kernel(z_target_reshaped, self.z_centers, self.z_bandwidth)
        Fz_target = Fz_target - np.mean(self._gaussian_kernel(self.z.reshape(-1, 1), 
                                                              self.z_centers, self.z_bandwidth), axis=0)
        
        f_target = Fz_target @ self.Bz @ self.final_a_vecs
        
        # Calcular corrección
        correction = np.zeros_like(y_samples)
        for k in range(len(self.final_sigma_vals)):
            if k < f_target.shape[1]:
                # Gradiente simplificado para funciones lineales y cuadráticas de y
                grad_g = np.array([1, 2 * y_samples]).T @ self.By @ self.final_b_vecs[:, k]
                correction += 2 * self.lambda_current * self.final_sigma_vals[k] * f_target[0, k] * grad_g
        
        x_samples = y_samples + correction
        
        return x_samples
    
    def estimate_conditional_density(self, x_eval, z_eval, bandwidth=None):
        """
        Estimar ρ(x|z) usando cambio de variables
        """
        # Implementación simplificada usando aproximación por kernels
        if bandwidth is None:
            bandwidth = np.std(self.x) / 5
        
        x_simulated = self.simulate_conditional(z_eval, n_samples=1000)
        
        # Estimación de densidad usando kernels gaussianos
        distances = ((x_simulated - x_eval) / bandwidth) ** 2
        density = np.mean(np.exp(-0.5 * distances)) / (bandwidth * np.sqrt(2 * np.pi))
        
        return density

# Ejemplo 9.1: Distribución gaussiana con media y varianza dependientes de z
def generate_example_data(n_samples=1500):
    """Generar datos del Ejemplo 9.1"""
    # z uniformemente distribuido en [-0.5, 0.5]^2
    z = np.random.uniform(-0.5, 0.5, size=(n_samples, 2))
    z1, z2 = z[:, 0], z[:, 1]
    
    # Parámetros dependientes de z
    mu_z = np.cos(2 * np.pi * z1) + np.sin(np.pi * z2)
    sigma_z = 0.2 * np.sqrt((1 - 2*z1) * (1 - 2*z2))
    
    # Generar muestras x
    x = np.random.normal(mu_z, sigma_z)
    
    return x, z

# Ejecutar ejemplo
if __name__ == "__main__":
    # Generar datos
    print("Generando datos del Ejemplo 9.1...")
    x, z = generate_example_data(1500)
    
    # Crear y ajustar modelo
    print("Ajustando modelo OTBP...")
    model = MongeOTBP(K_max=3)
    model.fit(x, z, max_iterations=500, verbose=True)
    
    # Visualizar resultados
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Datos originales
    axes[0, 0].scatter(z[:, 0], x, alpha=0.5, s=1)
    axes[0, 0].set_title('Datos Originales x vs z1')
    axes[0, 0].set_xlabel('z1')
    axes[0, 0].set_ylabel('x')
    
    # Barycenter
    axes[1, 0].hist(model.y_final, bins=50, alpha=0.7, density=True)
    axes[1, 0].set_title('Barycenter y')
    axes[1, 0].set_xlabel('y')
    axes[1, 0].set_ylabel('Densidad')
    
    # Simulación condicional para z* = (0.2, 0.3)
    z_star1 = np.array([0.2, 0.3])
    x_sim1 = model.simulate_conditional(z_star1, n_samples=500)
    
    axes[0, 1].hist(x_sim1, bins=30, alpha=0.7, density=True, label='Simulado')
    
    # Densidad verdadera para comparación
    mu_true1 = np.cos(2*np.pi*0.2) + np.sin(np.pi*0.3)
    sigma_true1 = 0.2 * np.sqrt((1-2*0.2) * (1-2*0.3))
    x_range1 = np.linspace(x_sim1.min(), x_sim1.max(), 100)
    true_density1 = (1/np.sqrt(2*np.pi*sigma_true1**2)) * np.exp(-0.5*((x_range1-mu_true1)/sigma_true1)**2)
    
    axes[0, 1].plot(x_range1, true_density1, 'r-', label='Verdadero', linewidth=2)
    axes[0, 1].set_title('ρ(x|z*) para z*=(0.2, 0.3)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Densidad')
    axes[0, 1].legend()
    
    # Simulación condicional para z* = (-0.2, -0.3)
    z_star2 = np.array([-0.2, -0.3])
    x_sim2 = model.simulate_conditional(z_star2, n_samples=500)
    
    axes[1, 1].hist(x_sim2, bins=30, alpha=0.7, density=True, label='Simulado')
    
    # Densidad verdadera para comparación
    mu_true2 = np.cos(2*np.pi*(-0.2)) + np.sin(np.pi*(-0.3))
    sigma_true2 = 0.2 * np.sqrt((1-2*(-0.2)) * (1-2*(-0.3)))
    x_range2 = np.linspace(x_sim2.min(), x_sim2.max(), 100)
    true_density2 = (1/np.sqrt(2*np.pi*sigma_true2**2)) * np.exp(-0.5*((x_range2-mu_true2)/sigma_true2)**2)
    
    axes[1, 1].plot(x_range2, true_density2, 'r-', label='Verdadero', linewidth=2)
    axes[1, 1].set_title('ρ(x|z*) para z*=(-0.2, -0.3)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].legend()
    
    # Convergencia
    history = model.history
    iterations = [h['iteration'] for h in history]
    losses = [h['total_loss'] for h in history]
    sigmas = [h['max_sigma'] for h in history]
    
    axes[0, 2].plot(iterations, losses)
    axes[0, 2].set_title('Convergencia: Pérdida Total')
    axes[0, 2].set_xlabel('Iteración')
    axes[0, 2].set_ylabel('Pérdida')
    axes[0, 2].set_yscale('log')
    
    axes[1, 2].plot(iterations, sigmas)
    axes[1, 2].axhline(y=model.nu/np.sqrt(len(x)), color='r', linestyle='--', 
                      label=f'Umbral σ* = {model.nu/np.sqrt(len(x)):.4f}')
    axes[1, 2].set_title('Convergencia: Valor Singular Máximo')
    axes[1, 2].set_xlabel('Iteración')
    axes[1, 2].set_ylabel('σ_max')
    axes[1, 2].legend()
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas finales
    print(f"\nResultados finales:")
    print(f"Valor singular máximo: {model.final_sigma_vals[0]:.6f}")
    print(f"Umbral σ*: {model.nu/np.sqrt(len(x)):.6f}")
    print(f"Lambda final: {model.lambda_current:.3f}")
    print(f"Número total de iteraciones: {len(model.history)}")
