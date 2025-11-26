'''  Clase de baricentro para uso mas transparente. Incluye la posibilidad de Gaussian o polionomial.

Tambien se agrega la distancia entre dos distribuciones a traves de muestras, puede ser copula o pueden ser cualquier muestra

'''


import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svd
from scipy.stats import spearmanr
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
#import matplotlib.pyplot as plt
import my_pyplot as plt
# Optimal transport using POT library
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("POT library not found. Install with: pip install POT")

def compute_dependence(y, z):
    """Calcula la dependencia entre y y z"""
    if z.ndim > 1:
        z = z.squeeze()
    return np.abs(spearmanr(y, z.squeeze())[0])

def gaussian_kernel(x, centers, bandwidth=0.1):
    """Calcula kernel gaussiano"""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)
    return np.exp(-np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2) / (2 * bandwidth**2))

def initialize_functional_spaces_gaussian(data, n_centers=10, bandwidth=0.1):
    """Inicializa espacio funcional usando kernel gaussiano"""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_centers, n_init=10).fit(data)
    centers = kmeans.cluster_centers_
    
    # Evaluar kernel
    F = gaussian_kernel(data, centers, bandwidth)
    
    # Centrar y SVD
    F_centered = F - np.mean(F, axis=0)
    U, S, Vt = svd(F_centered, full_matrices=False)
    Q = U @ np.diag(S)
    B = Vt.T / S
    
    return Q, B, centers


class OTSolverState:
    """Encapsula todo el estado del solver de transporte óptimo"""
    
    def __init__(self, method='gaussian'):
        """
        Parámetros:
        -----------
        method : str, {'gaussian', 'polynomial'}
            Método de representación funcional usado
        """
        self.method = method
        
        # Matrices de proyección
        self.Qz = None
        self.Bz = None
        self.Qy = None
        self.By = None
        
        # Matrices SVD
        self.U = None
        self.Vt = None
        self.s = None
        
        # Centros de kernels (solo para gaussian)
        self.centers_x = None
        self.centers_z = None
        
        # Parámetros
        self.lambda_val = None
        self.bandwidth_x = None
        self.bandwidth_z = None
    
    def simulate_conditional(self, y_samples, z_target):
        """
        Invierte el mapa T: X(y, z) = y + 2λ Σ_k σ_k ∇_y σ_k|_{y,z}
        
        Parámetros:
        -----------
        y_samples : array (n_samples,)
            Muestras del baricentro
        z_target : float
            Valor objetivo de Z
        
        Retorna:
        --------
        x_samples : array (n_samples,)
            Muestras condicionales X|Z=z_target
        """
        if self.method == 'gaussian':
            return self._simulate_conditional_gaussian(y_samples, z_target)
        elif self.method == 'polynomial':
            return self._simulate_conditional_polynomial(y_samples, z_target)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _simulate_conditional_gaussian(self, y_samples, z_target):
        """Simulación condicional con kernels gaussianos"""
        n_samples = len(y_samples)
        
        # 1. Calcular f_k(z_target)
        z_target_2d = np.array(z_target).reshape(1, -1)
        F_target = gaussian_kernel(z_target_2d, self.centers_z, self.bandwidth_z)
        F_target_centered = F_target - np.mean(F_target)
        Qz_target = F_target_centered @ self.Bz
        
        f_k_values = []
        for k in range(len(self.s)):
            a_k = self.U[:, k]
            f_k = float(Qz_target @ a_k)
            f_k_values.append(f_k)
        
        # 2. Calcular X(y_i, z_target)
        x_samples = np.zeros(n_samples)
        
        for i, y_val in enumerate(y_samples):
            x_i = y_val
            
            y_reshaped = np.array([[y_val]])
            K = gaussian_kernel(y_reshaped, self.centers_x, self.bandwidth_x)
            dK = -(y_reshaped - self.centers_x.T) / (self.bandwidth_x**2) * K
            
            for k in range(len(self.s)):
                b_k = self.Vt[k, :]
                grad_g_k = float(dK @ (self.By @ b_k))
                grad_sigma_k = f_k_values[k] * grad_g_k
                x_i += 2 * self.lambda_val * self.s[k] * grad_sigma_k
            
            x_samples[i] = x_i
        
        return x_samples
    
    def _simulate_conditional_polynomial(self, y_samples, z_target):
        """Simulación condicional con características polinomiales"""
        n_samples = len(y_samples)
        
        # 1. Calcular f_k(z_target)
        z_target_2d = np.array(z_target).reshape(1, -1)
        F_target = gaussian_kernel(z_target_2d, self.centers_z, self.bandwidth_z)
        F_target_centered = F_target - np.mean(F_target)
        Qz_target = F_target_centered @ self.Bz
        
        f_k_values = []
        for k in range(len(self.s)):
            a_k = self.U[:, k]
            f_k = float(Qz_target @ a_k)
            f_k_values.append(f_k)
        
        # 2. Calcular X(y_i, z_target) usando derivadas polinomiales
        x_samples = np.zeros(n_samples)
        
        for i, y_val in enumerate(y_samples):
            x_i = y_val
            for k in range(len(self.s)):
                b_k = self.Vt[k, :]
                # Para G = [y, y²], dG/dy = [1, 2y]
                grad_G = np.array([1.0, 2.0 * y_val])
                grad_g_k = float(grad_G @ (self.By @ b_k))
                grad_sigma_k = f_k_values[k] * grad_g_k
                x_i += 2 * self.lambda_val * self.s[k] * grad_sigma_k
            
            x_samples[i] = x_i
        
        return x_samples


def ot_barycenter_solver(x, z, method='gaussian', n_centers_x=10, n_centers_z=10, 
                         bandwidth_x=0.1, bandwidth_z=0.1, 
                         n_iter=1000, verbose=False, aggressive=False):
    """
    Resuelve el problema de baricentro OT usando kernels gaussianos para X y Z.
    
    Parámetros:
    -----------
    x : array (n,)
        Datos de entrada
    z : array (n,)
        Variable condicional
    n_centers_x : int
        Número de centros para kernel gaussiano en X
    n_centers_z : int
        Número de centros para kernel gaussiano en Z
    bandwidth_x : float
        Ancho de banda para kernel en X
    bandwidth_z : float
        Ancho de banda para kernel en Z
    n_iter : int
        Número máximo de iteraciones
    verbose : bool
        Imprimir información de progreso
    aggressive : bool
        Usar parámetros más agresivos para convergencia rápida (útil para problemas simples)
    
    Retorna:
    --------
    y : array (n,)
        Baricentro óptimo
    state : OTSolverState
        Objeto con todo el estado necesario para simulación condicional
    """
    n = len(x)
    y = x.copy()
    y0 = y.copy()
    
    # Inicializar el estado
    state = OTSolverState(method='gaussian')
    state.bandwidth_x = bandwidth_x
    state.bandwidth_z = bandwidth_z
    
    # Espacio funcional F para z (fijo)
    state.Qz, state.Bz, state.centers_z = initialize_functional_spaces_gaussian(
        z, n_centers=n_centers_z, bandwidth=bandwidth_z
    )
    
    # Parámetros
    sigma_star = 0.2 / np.sqrt(n)
    alpha = 0.0025
    delta = 0.1
    
    if aggressive:
        # Parámetros más agresivos para convergencia rápida
        theta = 1.0  # No aumentar eta
        tau = 0.8    # Reducir más lento
        kappa = 0.01 # Condición de Armijo MUY laxa
        eta_prev = 1e-1  # Eta inicial más grande
        sigma_star = 0.5 / np.sqrt(n)  # Umbral más alto
        eta_min = 1e-6  # Límite inferior para eta más alto
        eta_reset = 1e-2  # Valor para resetear eta cuando se estanca
    else:
        theta = 1.0  
        tau = 0.8
        kappa = 0.05
        eta_prev = 1e-2
        eta_min = 1e-8
        eta_reset = 1e-3
    
    # Espacio funcional G para y0 (inicial)
    Qy0, By0, centers_x0 = initialize_functional_spaces_gaussian(
        y0, n_centers=n_centers_x, bandwidth=bandwidth_x
    )
    state.centers_x = centers_x0.copy()
    
    # Contadores para detección de estancamiento
    stagnant_count = 0
    prev_dep = float('inf')
    
    for iter in range(n_iter):
        # Evaluar G(y) usando los centros fijos de esta etapa
        G_current = gaussian_kernel(y.reshape(-1, 1), state.centers_x, bandwidth_x)
        G_current_centered = G_current - np.mean(G_current, axis=0)
        Qy = G_current_centered @ By0
        
        # Matriz A y SVD
        A = state.Qz.T @ Qy / n
        U, s, Vt = svd(A, full_matrices=False)
        
        # Gradiente del costo de transporte
        cost_grad = y - x
        
        # Gradiente del término de penalización
        penalty_grad = np.zeros_like(y)
        
        for k in range(len(s)):
            a_k = U[:, k]
            b_k = Vt[k, :]
            f_k = state.Qz @ a_k
            
            y_reshaped = y.reshape(-1, 1)
            K = gaussian_kernel(y_reshaped, state.centers_x, bandwidth_x)
            dK = -(y_reshaped - state.centers_x.T) / (bandwidth_x**2) * K
            dg_k = (dK @ (By0 @ b_k)).squeeze()
            
            penalty_grad += 2 * s[k] * f_k * dg_k
        
        # Cálculo adaptativo de lambda
        num = np.sqrt(np.mean(cost_grad**2) + 0.1 * np.var(x))
        
        penalty_grad_tilde = np.zeros_like(y)
        for k in range(len(s)):
            sigma_tilde = min(s[k], sigma_star)
            a_k = U[:, k]
            b_k = Vt[k, :]
            f_k = state.Qz @ a_k
            
            y_reshaped = y.reshape(-1, 1)
            K = gaussian_kernel(y_reshaped, state.centers_x, bandwidth_x)
            dK = -(y_reshaped - state.centers_x.T) / (bandwidth_x**2) * K
            dg_k = (dK @ (By0 @ b_k)).squeeze()
            
            penalty_grad_tilde += sigma_tilde * f_k * dg_k
        
        denom = np.sqrt(np.mean(penalty_grad_tilde**2))
        lambda_val = 0.5 * num / (denom + 1e-12)
        
        # Gradiente total con clipping para estabilidad
        G = cost_grad + lambda_val * penalty_grad
        
        # Clip gradiente si es muy grande
        grad_norm = np.sqrt(np.mean(G**2))
        if grad_norm > 10.0:
            G = G * (10.0 / grad_norm)
            if verbose and iter % 100 == 0:
                print(f"  Gradient clipped from {grad_norm:.2f} to 10.0")
        
        # Backtracking line search con límite de iteraciones
        eta = max(theta * eta_prev, eta_min)  # No dejar que eta sea demasiado pequeño
        L_current = np.sum((y - x)**2) / (2*n) + lambda_val * np.sum(s**2)
        
        max_bt_iter = 30
        armijo_satisfied = False
        for bt_iter in range(max_bt_iter):
            y_new = y - eta * G
            
            G_new = gaussian_kernel(y_new.reshape(-1, 1), state.centers_x, bandwidth_x)
            G_new_centered = G_new - np.mean(G_new, axis=0)
            Qy_new = G_new_centered @ By0
            A_new = state.Qz.T @ Qy_new / n
            _, s_new, _ = svd(A_new, full_matrices=False)
            
            L_new = np.sum((y_new - x)**2) / (2*n) + lambda_val * np.sum(s_new**2)
            
            if L_new <= L_current - kappa * eta * np.mean(G**2) or eta <= eta_min:
                armijo_satisfied = True
                break
            eta *= tau
        
        if not armijo_satisfied and verbose and iter % 100 == 0:
            print(f"  Warning: Armijo not satisfied, using eta={eta:.3e}")
        
        # Detectar estancamiento y resetear eta
        current_dep = compute_dependence(y_new, z)
        if abs(current_dep - prev_dep) < 1e-6:
            stagnant_count += 1
        else:
            stagnant_count = 0
        prev_dep = current_dep
        
        if stagnant_count > 50 and eta < eta_reset:
            eta = eta_reset
            stagnant_count = 0
            if verbose:
                print(f"  → Resetting eta to {eta_reset:.3e} due to stagnation")
        
        eta_prev = eta
        y = y_new
        
        if verbose and iter % 100 == 0:
            dep = compute_dependence(y, z)
            print(f"Iter {iter}: dep={dep:.6f}, max_s={np.max(s):.6f}, "
                  f"lambda={lambda_val:.3e}, eta={eta:.3e}, ||G||²={np.mean(G**2):.3e}")
        
        # Criterio de reinicio de etapa
        if np.sum((y - y0)**2) > delta * np.sum((y0 - np.mean(y0))**2):
            if verbose:
                print(f"→ Reinicio de etapa en iter {iter}")
            y0 = y.copy()
            Qy0, By0, centers_x_new = initialize_functional_spaces_gaussian(
                y0, n_centers=n_centers_x, bandwidth=bandwidth_x
            )
            state.centers_x = centers_x_new
        
        # Criterio de convergencia
        if np.all(s < sigma_star):
            grad_norm = np.mean(G**2)
            if grad_norm < alpha * np.var(x):
                if verbose:
                    print(f"Convergió en iteración {iter}")
                break
    
    # Guardar estado final
    state.U = U
    state.Vt = Vt
    state.s = s
    state.Qy = Qy
    state.By = By0
    state.lambda_val = lambda_val
    
    return y, state


def _ot_barycenter_solver_polynomial(x, z, n_centers_z=10, bandwidth_z=0.1,
                                     n_iter=1000, verbose=False, aggressive=False):
    """Implementación interna con características polinomiales [y, y²]"""
    n = len(x)
    y = x.copy()
    y0 = y.copy()
    
    # Inicializar el estado
    state = OTSolverState(method='polynomial')
    state.bandwidth_z = bandwidth_z
    
    # Espacio funcional F para z (fijo, kernel gaussiano)
    state.Qz, state.Bz, state.centers_z = initialize_functional_spaces_gaussian(
        z, n_centers=n_centers_z, bandwidth=bandwidth_z
    )
    
    # Parámetros
    sigma_star = 0.2 / np.sqrt(n)
    alpha = 0.0025
    delta = 0.1
    
    if aggressive:
        theta = 1.0
        tau = 0.8
        kappa = 0.01
        eta_prev = 1e-1
        sigma_star = 0.5 / np.sqrt(n)
        eta_min = 1e-6
        eta_reset = 1e-2
    else:
        theta = 1.0
        tau = 0.8
        kappa = 0.05
        eta_prev = 1e-2
        eta_min = 1e-8
        eta_reset = 1e-3
    
    # Espacio funcional G para y0: características polinomiales
    G0 = np.column_stack([y0, y0**2])
    G0_centered = G0 - np.mean(G0, axis=0)
    Uy0, Sy0, Vty0 = svd(G0_centered, full_matrices=False)
    By0 = Vty0.T / Sy0
    
    for iter in range(n_iter):
        # Evaluar G en los y actuales pero con By fijo
        G_current = np.column_stack([y, y**2])
        G_current_centered = G_current - np.mean(G0, axis=0)
        Qy = G_current_centered @ By0
        
        # Matriz A y SVD
        A = state.Qz.T @ Qy / n
        U, s, Vt = svd(A, full_matrices=False)
        
        # Gradiente del costo
        cost_grad = y - x
        
        # Gradiente del término de penalización
        penalty_grad = np.zeros_like(y)
        for k in range(len(s)):
            a_k = U[:, k]
            b_k = Vt[k, :]
            f_k = state.Qz @ a_k
            dG = np.column_stack([np.ones_like(y), 2*y])
            dg_k = dG @ (By0 @ b_k)
            penalty_grad += 2 * s[k] * f_k * dg_k
        
        # Cálculo adaptativo de lambda
        num = np.sqrt(np.mean(cost_grad**2) + 0.1 * np.var(x))
        
        penalty_grad_tilde = np.zeros_like(y)
        for k in range(len(s)):
            sigma_tilde = min(s[k], sigma_star)
            a_k = U[:, k]
            b_k = Vt[k, :]
            f_k = state.Qz @ a_k
            dG = np.column_stack([np.ones_like(y), 2*y])
            dg_k = dG @ (By0 @ b_k)
            penalty_grad_tilde += sigma_tilde * f_k * dg_k
        
        denom = np.sqrt(np.mean(penalty_grad_tilde**2))
        lambda_val = 0.5 * num / (denom + 1e-12)
        
        # Gradiente total
        G = cost_grad + lambda_val * penalty_grad
        
        # Backtracking line search con límite
        eta = max(theta * eta_prev, eta_min)
        L_current = np.sum((y - x)**2) / (2*n) + lambda_val * np.sum(s**2)
        
        max_bt_iter = 30
        armijo_satisfied = False
        for bt_iter in range(max_bt_iter):
            y_new = y - eta * G
            G_new = np.column_stack([y_new, y_new**2])
            G_new_centered = G_new - np.mean(G0, axis=0)
            Qy_new = G_new_centered @ By0
            A_new = state.Qz.T @ Qy_new / n
            _, s_new, _ = svd(A_new, full_matrices=False)
            L_new = np.sum((y_new - x)**2) / (2*n) + lambda_val * np.sum(s_new**2)
            if L_new <= L_current - kappa * eta * np.mean(G**2) or eta <= eta_min:
                armijo_satisfied = True
                break
            eta *= tau
        
        if not armijo_satisfied and verbose and iter % 100 == 0:
            print(f"  Warning: Armijo not satisfied, using eta={eta:.3e}")
        
        eta_prev = eta
        y = y_new
        
        if verbose and iter % 100 == 0:
            dep = compute_dependence(y, z)
            print(f"Iter {iter}: dep={dep:.6f}, max_s={np.max(s):.6f}, "
                  f"lambda={lambda_val:.3e}, eta={eta:.3e}, ||G||²={np.mean(G**2):.3e}")
        
        # Criterio de reinicio de etapa
        if np.sum((y - y0)**2) > delta * np.sum((y0 - np.mean(y0))**2):
            if verbose:
                print(f"→ Reinicio de etapa en iter {iter}")
            y0 = y.copy()
            G0 = np.column_stack([y0, y0**2])
            G0_centered = G0 - np.mean(G0, axis=0)
            Uy0, Sy0, Vty0 = svd(G0_centered, full_matrices=False)
            By0 = Vty0.T / Sy0
        
        # Criterio de convergencia
        if np.all(s < sigma_star):
            grad_norm = np.mean(G**2)
            if grad_norm < alpha * np.var(x):
                if verbose:
                    print(f"Convergió en iteración {iter}")
                break
    
    # Guardar estado final
    state.U = U
    state.Vt = Vt
    state.s = s
    state.Qy = Qy
    state.By = By0
    state.lambda_val = lambda_val
    
    return y, state


def interpolate_barycenter(z_current, cached_Z, cached_y):
    """Interpolación robusta para baricentros"""
    if len(cached_Z) < 2:
        return cached_y[0] if len(cached_y) > 0 else np.nan
    
    sort_idx = np.argsort(cached_Z)
    Z_sorted = cached_Z[sort_idx]
    y_sorted = cached_y[sort_idx]
    
    if z_current <= Z_sorted[0]:
        return y_sorted[0]
    elif z_current >= Z_sorted[-1]:
        return y_sorted[-1]
    else:
        return np.interp(z_current, Z_sorted, y_sorted)


def wasserstein_distance(samples, target):
    """Compute 2-Wasserstein distance between two samples"""
    n1 = len(samples)
    n2 = len(target)
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2
    
    M = ot.dist(samples, target, metric='euclidean')
    M = M ** 2
    
    W2_squared = ot.emd2(a, b, M)
    return np.sqrt(W2_squared)

class CopulaFitter:
    """
    Fits various copulas to bivariate data using Wasserstein distance
    """
    
    def __init__(self, data):
        """
        Parameters:
        -----------
        data : array-like, shape (n, 2)
            Bivariate data to fit
        """
        self.data = np.array(data)
        self.n = len(self.data)
        
        # Transform to pseudo-observations (empirical copula)
        self.uniform_data = self._to_uniform_marginals(self.data)
    
    def _to_uniform_marginals(self, data):
        """Transform data to uniform marginals using empirical CDF"""
        n = len(data)
        uniform = np.zeros_like(data)
        for j in range(data.shape[1]):
            ranks = stats.rankdata(data[:, j])
            uniform[:, j] = ranks / (n + 1)
        return uniform
    
    def _gaussian_copula_sample(self, n_samples, rho):
        """Generate samples from Gaussian copula"""
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        # Transform to uniform using standard normal CDF
        uniform_samples = stats.norm.cdf(samples)
        return uniform_samples
    
    def _clayton_copula_sample(self, n_samples, theta):
        """Generate samples from Clayton copula"""
        if theta <= 0:
            theta = 0.01  # Avoid numerical issues
        
        u = np.random.uniform(0, 1, n_samples)
        v_samples = np.random.uniform(0, 1, n_samples)
        
        # Conditional sampling
        v = (u**(-theta) * (v_samples**(-theta/(1+theta)) - 1) + 1)**(-1/theta)
        
        return np.column_stack([u, v])
    
    def _gumbel_copula_sample(self, n_samples, theta):
        """Generate samples from Gumbel copula"""
        if theta < 1:
            theta = 1.01  # Gumbel parameter must be >= 1
        
        # Using the algorithm from Hofert (2008)
        alpha = 1 / theta
        u = np.random.uniform(0, 1, n_samples)
        
        # Generate from stable distribution
        v = np.random.uniform(0, 1, n_samples)
        w = np.random.exponential(1, n_samples)
        
        s = self._stable_sample(alpha, n_samples)
        
        u1 = np.exp(-(-np.log(u) / s)**(1/theta))
        u2 = np.exp(-(-np.log(v) / s)**(1/theta))
        
        return np.column_stack([u1, u2])
    
    def _stable_sample(self, alpha, n):
        """Sample from stable distribution (simplified)"""
        # Simplified stable distribution sampling
        return np.random.gamma(1/alpha, 1, n)
    
    def _frank_copula_sample(self, n_samples, theta):
        """Generate samples from Frank copula"""
        if abs(theta) < 0.01:
            # Return independence copula for theta near 0
            return np.random.uniform(0, 1, (n_samples, 2))
        
        u = np.random.uniform(0, 1, n_samples)
        t = np.random.uniform(0, 1, n_samples)
        
        # Conditional sampling with numerical stability
        if theta > 0:
            # Positive dependence
            exp_theta = np.exp(-theta)
            exp_theta_u = np.exp(-theta * u)
            
            numerator = -np.log(1 - t * (1 - exp_theta))
            denominator = theta + np.log(t * exp_theta_u + (1 - t))
            
            v = numerator / denominator
        else:
            # Negative dependence
            theta_abs = abs(theta)
            exp_theta = np.exp(theta_abs)
            exp_theta_u = np.exp(theta_abs * u)
            
            numerator = np.log(1 + t * (exp_theta - 1))
            denominator = theta_abs - np.log((1 - t) + t * exp_theta_u)
            
            v = numerator / denominator
        
        # Clip to valid range to avoid numerical errors
        v = np.clip(v, 1e-10, 1 - 1e-10)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        
        return np.column_stack([u, v])
    
    def _independence_copula_sample(self, n_samples):
        """Generate samples from independence copula"""
        return np.random.uniform(0, 1, (n_samples, 2))
    
    def wasserstein_distance(self, copula_samples):
        """
        Compute 2-Wasserstein distance between data and copula samples
        """
        if not HAS_POT:
            # Fallback to simple distance measure if POT not available
            return self._approximate_wasserstein(copula_samples)
        
        # Uniform weights
        a = np.ones(self.n) / self.n
        b = np.ones(len(copula_samples)) / len(copula_samples)
        
        # Cost matrix (Euclidean distance squared)
        M = ot.dist(self.uniform_data, copula_samples, metric='euclidean')
        M = M ** 2  # Squared for W2 distance
        
        # Compute optimal transport
        W2_squared = ot.emd2(a, b, M)
        
        return np.sqrt(W2_squared)
    
    def _approximate_wasserstein(self, copula_samples):
        """Approximate Wasserstein distance without POT library"""
        # Simple approximation: mean of minimum distances
        from scipy.spatial.distance import cdist
        
        dist_matrix = cdist(self.uniform_data, copula_samples, metric='euclidean')
        min_distances = np.min(dist_matrix, axis=1)
        
        return np.mean(min_distances)
    
    def fit_copula(self, copula_type, param_range=None, n_samples=None):
        """
        Fit a copula by minimizing Wasserstein distance
        
        Parameters:
        -----------
        copula_type : str
            Type of copula: 'gaussian', 'clayton', 'gumbel', 'frank', 'independence'
        param_range : tuple
            Range for parameter search (min, max)
        n_samples : int
            Number of samples to generate from copula (default: same as data)
        
        Returns:
        --------
        dict : Best parameter and Wasserstein distance
        """
        if n_samples is None:
            n_samples = self.n
        
        if copula_type == 'independence':
            samples = self._independence_copula_sample(n_samples)
            distance = self.wasserstein_distance(samples)
            return {'copula': 'independence', 'parameter': None, 'distance': distance}
        
        # Set default parameter ranges
        if param_range is None:
            if copula_type == 'gaussian':
                param_range = (-0.99, 0.99)
            elif copula_type == 'clayton':
                param_range = (0.1, 10)
            elif copula_type == 'gumbel':
                param_range = (1.01, 10)
            elif copula_type == 'frank':
                param_range = (-10, 10)
        
        def objective(param):
            param = param[0]
            try:
                if copula_type == 'gaussian':
                    samples = self._gaussian_copula_sample(n_samples, param)
                elif copula_type == 'clayton':
                    samples = self._clayton_copula_sample(n_samples, max(0.01, param))
                elif copula_type == 'gumbel':
                    samples = self._gumbel_copula_sample(n_samples, max(1.01, param))
                elif copula_type == 'frank':
                    samples = self._frank_copula_sample(n_samples, param)
                else:
                    raise ValueError(f"Unknown copula type: {copula_type}")
                
                return self.wasserstein_distance(samples)
            except:
                return 1e10  # Large penalty for invalid parameters
        
        # Optimize
        result = minimize(objective, 
                         x0=[(param_range[0] + param_range[1]) / 2],
                         bounds=[param_range],
                         method='L-BFGS-B')
        
        return {
            'copula': copula_type,
            'parameter': result.x[0],
            'distance': result.fun
        }
    
    def fit_all_copulas(self, n_samples=None):
        """
        Fit all available copulas and return comparison
        """
        copulas = ['gaussian', 'clayton', 'gumbel', 'frank', 'independence']
        results = []
        
        print("Fitting copulas using Wasserstein distance...\n")
        
        for copula in copulas:
            print(f"Fitting {copula} copula...")
            result = self.fit_copula(copula, n_samples=n_samples)
            results.append(result)
            print(f"  Parameter: {result['parameter']}")
            print(f"  W2 Distance: {result['distance']:.6f}\n")
        
        # Sort by distance
        results.sort(key=lambda x: x['distance'])
        
        return results
    
    def plot_comparison(self, results, n_samples=1000):
        """
        Plot original data and best fitting copulas
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot original data
        axes[0].scatter(self.uniform_data[:, 0], self.uniform_data[:, 1], 
                       alpha=0.5, s=20, label='Data')
        axes[0].set_title('Original Data (Uniform Marginals)')
        axes[0].set_xlabel('U')
        axes[0].set_ylabel('V')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot top 5 copulas
        for idx, result in enumerate(results[:5]):
            copula = result['copula']
            param = result['parameter']
            distance = result['distance']
            
            # Generate samples
            if copula == 'gaussian':
                samples = self._gaussian_copula_sample(n_samples, param)
            elif copula == 'clayton':
                samples = self._clayton_copula_sample(n_samples, param)
            elif copula == 'gumbel':
                samples = self._gumbel_copula_sample(n_samples, param)
            elif copula == 'frank':
                samples = self._frank_copula_sample(n_samples, param)
            elif copula == 'independence':
                samples = self._independence_copula_sample(n_samples)
            
            axes[idx + 1].scatter(samples[:, 0], samples[:, 1], 
                                 alpha=0.5, s=20, c='orange')
            
            title = f'{copula.capitalize()} Copula'
            if param is not None:
                title += f'\nθ={param:.3f}'
            title += f'\nW₂={distance:.4f}'
            
            axes[idx + 1].set_title(title)
            axes[idx + 1].set_xlabel('U')
            axes[idx + 1].set_ylabel('V')
            axes[idx + 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tmp/copulas.png')


# Example usage
if __name__ == "__main__":
    # Generate sample data from a known copula (for testing)
    np.random.seed(42)
    
    # True Gaussian copula with rho=0.7
    true_rho = 0.7
    mean = [0, 0]
    cov = [[1, true_rho], [true_rho, 1]]
    samples = np.random.multivariate_normal(mean, cov, 500)
    data = stats.norm.cdf(samples)
    
    # Fit copulas
    fitter = CopulaFitter(data)
    results = fitter.fit_all_copulas()
    
    # Print summary
    print("\n" + "="*60)
    print("RANKING OF COPULAS BY WASSERSTEIN DISTANCE")
    print("="*60)
    for i, result in enumerate(results, 1):
        param_str = f"θ={result['parameter']:.4f}" if result['parameter'] is not None else "N/A"
        print(f"{i}. {result['copula'].capitalize():15s} | {param_str:15s} | W₂={result['distance']:.6f}")
    
    # Plot comparison
    fitter.plot_comparison(results)
    
