import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import svd
from scipy.stats import spearmanr

###

def compute_dependence(y, z):
    """Calcula la dependencia entre y y z"""
    if z.ndim > 1:
        z = z.squeeze()
        return np.abs(spearmanr(y, z)[0])
    return np.abs(spearmanr(y, z.squeeze())[0])

def gaussian_kernel(x, centers, bandwidth=0.1):
    """Calcula kernel gaussiano"""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)
    return np.exp(-np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2) / (2 * bandwidth**2))

def initialize_functional_spaces(z, n_centers=10, bandwidth=0.1):
    """Inicializa el espacio funcional F para z"""
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    kmeans_z = KMeans(n_clusters=n_centers, n_init=10).fit(z)
    centers_z = kmeans_z.cluster_centers_
    F = gaussian_kernel(z, centers_z, bandwidth)
    
    F_centered = F - np.mean(F, axis=0)
    Uz, Sz, Vtz = svd(F_centered, full_matrices=False)
    Qz = Uz @ np.diag(Sz)
    Bz = Vtz.T / Sz
    
    return Qz, Bz, centers_z

def compute_penalty_gradient(y, Qz):
    """Calcula el gradiente del término de penalización """
    n = len(y)
    
    # Recalcula todo con los y actuales
    G_current = np.column_stack([y, y**2])
    G_current_centered = G_current - np.mean(G_current, axis=0)
    Uy_current, Sy_current, Vty_current = svd(G_current_centered, full_matrices=False)
    By_current = Vty_current.T / Sy_current
    
    Qy_current = G_current_centered @ By_current
    A = Qz.T @ Qy_current / n
    U, s, Vt = svd(A, full_matrices=False)
    
    penalty_grad = np.zeros_like(y)
    
    for k in range(len(s)):
        a_k = U[:, k]
        b_k = Vt[k, :]
        
        f_k = Qz @ a_k
        dG = np.column_stack([np.ones_like(y), 2 * y])
        dg_k = dG @ (By_current @ b_k)
        
        penalty_grad += 2 * s[k] * f_k * dg_k
    
    return penalty_grad, s, U, Vt, By_current

def ot_barycenter_solver(x, z, n_iter=1000, verbose=False):
    n = len(x)
    y = x.copy()
    y0 = y.copy()  # Estado fijo de la etapa
    
    # Espacio funcional F para z (fijo)
    Qz, Bz, centers_z = initialize_functional_spaces(z)
    
    # Parámetros
    sigma_star = 0.2 / np.sqrt(n)
    alpha = 0.0025
    delta = 0.1
    
    theta = 1.1
    tau = 0.5
    kappa = 0.5
    eta_prev = 1e-2
    
    # Espacio funcional G fijo para esta etapa
    G0 = np.column_stack([y0, y0**2])
    G0_centered = G0 - np.mean(G0, axis=0)
    Uy0, Sy0, Vty0 = svd(G0_centered, full_matrices=False)
    By0 = Vty0.T / Sy0  # FIJO durante la etapa
    
    for iter in range(n_iter):
        # Evaluar G en los y actuales pero con By fijo
        G_current = np.column_stack([y, y**2])
        G_current_centered = G_current - np.mean(G0, axis=0)  # Centrar con media de y0
        Qy = G_current_centered @ By0
        
        # Matriz A y SVD
        A = Qz.T @ Qy / n
        U, s, Vt = svd(A, full_matrices=False)
        cost_grad = y - x
        penalty_grad = np.zeros_like(y)
        
        for k in range(len(s)):
            a_k = U[:, k]
            b_k = Vt[k, :]
            f_k = Qz @ a_k
            dG = np.column_stack([np.ones_like(y), 2*y])
            dg_k = dG @ (By0 @ b_k)
            penalty_grad += 2 * s[k] * f_k * dg_k
        
        num = np.sqrt(np.mean(cost_grad**2) + 0.1 * np.var(x))
        penalty_grad_tilde = np.zeros_like(y)

        for k in range(len(s)):
            sigma_tilde = min(s[k], sigma_star)
            a_k = U[:, k]
            b_k = Vt[k, :]
            f_k = Qz @ a_k
            dG = np.column_stack([np.ones_like(y), 2*y])
            dg_k = dG @ (By0 @ b_k)
            penalty_grad_tilde += sigma_tilde * f_k * dg_k
        
        denom = np.sqrt(np.mean(penalty_grad_tilde**2))
        lambda_val = 0.5 * num / (denom + 1e-12)
        G = cost_grad + lambda_val * penalty_grad
        eta = theta * eta_prev
        L_current = np.sum((y - x)**2) / (2*n) + lambda_val * np.sum(s**2)
        
        while True:
            y_new = y - eta * G
            G_new = np.column_stack([y_new, y_new**2])
            G_new_centered = G_new - np.mean(G0, axis=0)
            Qy_new = G_new_centered @ By0
            A_new = Qz.T @ Qy_new / n
            _, s_new, _ = svd(A_new, full_matrices=False)
            L_new = np.sum((y_new - x)**2) / (2*n) + lambda_val * np.sum(s_new**2)
            if L_new <= L_current - kappa * eta * np.mean(G**2):
                break
            eta *= tau
        
        eta_prev = eta
        y = y_new
        
        if verbose and iter % 1000 == 0:
            dep = compute_dependence(y, z)
            print(f"Iter {iter}: dep={dep:.6f}, max_s={np.max(s):.6f}, "
                  f"lambda={lambda_val:.3e}, eta={eta:.3e}")
        
        # Criterio de reinicio de etapa
        if np.sum((y - y0)**2) > delta * np.sum((y0 - np.mean(y0))**2):
            if verbose:
                print(f"→ Reinicio de etapa en iter {iter}")
            y0 = y.copy()
            G0 = np.column_stack([y0, y0**2])
            G0_centered = G0 - np.mean(G0, axis=0)
            Uy0, Sy0, Vty0 = svd(G0_centered, full_matrices=False)
            By0 = Vty0.T / Sy0
        
        if np.all(s < sigma_star):
            grad_norm = np.mean(G**2)
            if grad_norm < alpha * np.var(x):
                if verbose:
                    print(f"Convergió en iteración {iter}")
                break
    
    return y, s, U, Vt, Qz, Bz, By0, centers_z, lambda_val


def simulate_conditional(y_samples, z_target, Qz, Bz, By, centers_z, lambda_final, s, U, Vt, bandwidth=0.1):
    """
    Invierte el mapa T usando la fórmula de la Sección 5:
    X(y, z) = y + 2λ Σ_k σ_k ∇_y σ_k|_{y,z}
    donde ∇_y σ_k = f_k(z) ∇g_k(y)
    """
    n_samples = len(y_samples)
    
    # 1. Calcular f_k(z_target) para todos los k
    # Primero evaluar F(z_target) en el espacio funcional
    z_target_2d = np.array(z_target).reshape(1, -1)
    F_target = gaussian_kernel(z_target_2d, centers_z, bandwidth)
    
    # Centrar con la misma media usada en el entrenamiento
    F_mean = np.mean(gaussian_kernel(centers_z, centers_z, bandwidth), axis=0)
    F_target_centered = F_target - F_mean
    
    # Proyectar al espacio ortogonal
    Qz_target = F_target_centered @ Bz  #[1, n_z]
    
    f_k_values = []
    for k in range(len(s)):
        a_k = U[:, k]  # [n_z,]
        f_k = float(Qz_target @ a_k) 
        f_k_values.append(f_k)
    
    # calcula X(y_i, z_target) para cada y_i
    x_samples = np.zeros(n_samples)
    
    for i, y_val in enumerate(y_samples):
        x_i = y_val
        for k in range(len(s)):
            b_k = Vt[k, :]  # shape: (n_y,)
            grad_G = np.array([1.0, 2.0 * y_val])  # shape: (2,)
            grad_g_k = float(grad_G @ (By @ b_k))  # escalar
            grad_sigma_k = f_k_values[k] * grad_g_k
            x_i += 2 * lambda_final * s[k] * grad_sigma_k
        
        x_samples[i] = x_i
    
    return x_samples

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