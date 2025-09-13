import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import svd

def generate_data(n=1500):
    """Genera datos para el ejemplo 9.1"""
    z = np.random.uniform(-0.5, 0.5, size=(n, 2))
    mu_z = np.cos(2 * np.pi * z[:, 0]) + np.sin(np.pi * z[:, 1])
    sigma_z = 0.2 * np.sqrt((1 - 2 * z[:, 0]) * (1 - 2 * z[:, 1]))
    x = np.random.normal(mu_z, sigma_z)
    return x, z

def gaussian_kernel(x, centers, bandwidth=0.1):
    """Calcula kernel gaussiano"""
    return np.exp(-np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2) / (2 * bandwidth**2))

def initialize_functional_spaces(x, z, n_centers=10, bandwidth=0.1):
    """Inicializa los espacios funcionales F y G"""
    # Espacio F para z (usando kernels gaussianos)
    kmeans_z = KMeans(n_clusters=n_centers, n_init=10).fit(z)
    centers_z = kmeans_z.cluster_centers_
    F = gaussian_kernel(z, centers_z, bandwidth)
    
    # Centrar y ortogonalizar F
    F_centered = F - np.mean(F, axis=0)
    Uz, Sz, Vtz = svd(F_centered, full_matrices=False)
    Qz = Uz @ np.diag(Sz)
    Bz = Vtz.T / Sz
    
    # Espacio G para y (funciones lineales y cuadráticas)
    G = np.column_stack([x, x**2])
    G_centered = G - np.mean(G, axis=0)
    Uy, Sy, Vty = svd(G_centered, full_matrices=False)
    Qy = Uy @ np.diag(Sy)
    By = Vty.T / Sy
    
    return Qz, Bz, Qy, By, centers_z

def compute_penalty_gradient(y, Qz, Bz, By, sigma, U, Vt):
    """Calcula el gradiente del término de penalización"""
    n = len(y)
    n_z = Qz.shape[1]
    
    # Recalcular G y Qy para los y actuales
    G_current = np.column_stack([y, y**2])
    Qy_current = G_current @ By
    
    # Calcular matriz A y su SVD
    A = Qz.T @ Qy_current
    U, s, Vt = svd(A, full_matrices=False)
    
    # Calcular gradientes para cada punto
    penalty_grad = np.zeros_like(y)
    for k in range(len(s)):
        a_k = U[:, k]
        b_k = Vt[k, :]
        
        # f_k(z_i) = Qz_i · a_k
        f_k = Qz @ a_k
        
        # ∇g_k(y_i) = [1, 2y_i] · By · b_k
        dG = np.column_stack([np.ones_like(y), 2*y])
        dg_k = dG @ (By @ b_k)
        
        penalty_grad += 2 * s[k] * f_k * dg_k
    
    return penalty_grad, s

def ot_barycenter_solver(x, z, lambda_val=1.0, n_iter=100, lr=0.01):
    """Implementa el solver de baricentro de transporte óptimo"""
    n = len(x)
    y = x.copy()
    
    # Inicializar espacios funcionales
    Qz, Bz, Qy_init, By, centers_z = initialize_functional_spaces(x, z)
    
    # Almacenar valores iniciales para el espacio funcional G
    y0 = x.copy()
    
    # Iterar
    for iter in range(n_iter):
        # Calcular gradiente del coste de transporte
        cost_grad = y - x
        
        # Calcular gradiente de penalización
        penalty_grad, s = compute_penalty_gradient(y, Qz, Bz, By, sigma=None, U=None, Vt=None)
        
        # Actualizar y
        y = y - lr * (cost_grad + lambda_val * penalty_grad / n)
        
        # Verificar criterio de convergencia (simplificado)
        if np.max(s) < 0.1/np.sqrt(n):
            break
    
    return y, s

def simulate_conditional(y_final, z_target, Qz, Bz, By, centers_z, lambda_val, s, U, Vt, bandwidth=0.1):
    """Simula la distribución condicional para z_target"""
    # Calcular F(z_target)
    F_target = gaussian_kernel(z_target.reshape(1, -1), centers_z, bandwidth)
    F_target_centered = F_target - np.mean(F_target, axis=0)
    Qz_target = F_target_centered @ Bz
    
    # Calcular x* para cada y en el baricentro
    x_sim = np.zeros_like(y_final)
    for i, y_val in enumerate(y_final):
        for k in range(len(s)):
            a_k = U[:, k]
            b_k = Vt[k, :]
            
            f_k = Qz_target @ a_k
            dG = np.array([1, 2*y_val])
            dg_k = dG @ (By @ b_k)
            
            x_sim[i] += 2 * lambda_val * s[k] * f_k * dg_k
        x_sim[i] += y_val
    
    return x_sim

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos
    np.random.seed(42)
    x, z = generate_data(1500)
    
    # Resolver problema de baricentro
    y_final, s = ot_barycenter_solver(x, z, lambda_val=0.1, n_iter=200, lr=0.001)
    
    # Simular para z_target
    z_target1 = np.array([0.2, 0.3])
    z_target2 = np.array([-0.2, -0.3])
    
    # Para simular, necesitamos recomputar algunos valores
    Qz, Bz, _, By, centers_z = initialize_functional_spaces(x, z)
    G_current = np.column_stack([y_final, y_final**2])
    Qy_current = G_current @ By
    A = Qz.T @ Qy_current
    U, s, Vt = svd(A, full_matrices=False)
    
    x_sim1 = simulate_conditional(y_final, z_target1, Qz, Bz, By, centers_z, 0.1, s, U, Vt)
    x_sim2 = simulate_conditional(y_final, z_target2, Qz, Bz, By, centers_z, 0.1, s, U, Vt)
    
    # Visualizar resultados
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.scatter(z[:, 0], z[:, 1], c=x, cmap='viridis', alpha=0.5)
    plt.colorbar(label='x')
    plt.title('Datos originales (x vs z)')
    plt.xlabel('z1')
    plt.ylabel('z2')
    
    plt.subplot(132)
    plt.hist(x_sim2, bins=30, alpha=0.7, density=True, label='z*=(-0.2,-0.3)')
    plt.hist(x_sim1, bins=30, alpha=0.4, density=True, label='z*=(0.2,0.3)')
    plt.legend()
    plt.title('Distribuciones condicionales simuladas')
    
    plt.subplot(133)
    plt.scatter(x, y_final, alpha=0.5)
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--')
    plt.xlabel('x original')
    plt.ylabel('y baricentro')
    plt.title('Relación entre x e y')
    
    plt.tight_layout()
    plt.show()
