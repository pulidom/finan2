import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svd
from scipy.stats import spearmanr

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
    
