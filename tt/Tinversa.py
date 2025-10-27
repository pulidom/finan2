def inverse_map(y_samples, z_target, Qz, Bz, By, centers_z, U, Vt, s, lambda_val, bandwidth=0.1):
    """
    Calcula la inversa T^{-1} para obtener muestras de x dado z_target
    
    Parameters:
    - y_samples: muestras del baricentro (y_i)
    - z_target: valor objetivo de z (puede ser escalar o vector)
    - Qz, Bz, By: matrices del espacio funcional
    - U, Vt, s: componentes SVD de la matriz A
    - lambda_val: parámetro de penalización
    - bandwidth: ancho de banda para el kernel
    
    Returns:
    - x_samples: muestras de x para z_target
    - f_k_values: valores de las funciones f_k(z_target) extraídas
    """
    
    n = len(y_samples)
    
    # Calcular f_k(z_target) para cada componente
    K_target = gaussian_kernel(np.array([z_target]).reshape(1, -1), centers_z, bandwidth)
    F_target = K_target
    F_target_centered = F_target - np.mean(F_target, axis=0)
    
    f_k_values = []
    for k in range(len(s)):
        a_k = U[:, k]
        f_k_z = F_target_centered @ (Bz @ a_k)
        f_k_values.append(f_k_z[0])
    
    # Calcular ∇g_k(y) para cada muestra y
    grad_g_k_values = []
    for i, y_val in enumerate(y_samples):
        dG = np.array([[1, 2 * y_val]])  # Para funciones [y, y^2]
        grad_g_k_sample = []
        for k in range(len(s)):
            b_k = Vt[k, :]
            dg_k = dG @ (By @ b_k)
            grad_g_k_sample.append(dg_k[0, 0])  # Asumiendo y unidimensional
        grad_g_k_values.append(grad_g_k_sample)
    
    grad_g_k_values = np.array(grad_g_k_values)  # shape: (n, K)
    
    # Aplicar fórmula de inversión (5.1)
    x_samples = np.zeros_like(y_samples)
    for i, y_val in enumerate(y_samples):
        correction = 0
        for k in range(len(s)):
            correction += s[k] * f_k_values[k] * grad_g_k_values[i, k]
        
        x_samples[i] = y_val + 2 * lambda_val * correction
    
    return x_samples, f_k_values

# Función auxiliar para simular ρ(x|z*)
def simulate_conditional_distribution(x_data, z_data, z_target, n_iter=1000, verbose=False):
    """
    Simula la distribución condicional ρ(x|z*) resolviendo el problema del baricentro
    y aplicando la inversa
    """
    # Resolver el problema del baricentro
    y_bary, s, U, Vt, Qz, Bz, By, centers_z, lambda_val = ot_barycenter_solver(
        x_data, z_data, n_iter=n_iter, verbose=verbose
    )
    
    # Obtener muestras de x para z_target
    x_samples, f_k_values = inverse_map(
        y_bary, z_target, Qz, Bz, By, centers_z, U, Vt, s, lambda_val
    )
    
    return x_samples, y_bary, f_k_values

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo (similar a los ejemplos del artículo)
    np.random.seed(42)
    n_samples = 1000
    
    # z uniforme en [-2, 2]
    z_data = np.random.uniform(-2, 2, n_samples)
    
    # x depende de z: x ~ N(z^2, 0.5^2) + algo de ruido
    x_data = z_data**2 + np.random.normal(0, 0.5, n_samples)
    
    # z objetivo
    z_target = 1.5
    
    # Simular ρ(x|z_target)
    x_samples, y_bary, f_k_values = simulate_conditional_distribution(
        x_data, z_data, z_target, n_iter=2000, verbose=True
    )
    
    print(f"\nResultados para z_target = {z_target}:")
    print(f"Muestras de x generadas: {len(x_samples)}")
    print(f"Media de x|z*: {np.mean(x_samples):.3f}")
    print(f"Desviación estándar de x|z*: {np.std(x_samples):.3f}")
    print(f"Valores de f_k(z_target): {[f'{f:.3f}' for f in f_k_values]}")
    
    # Visualización
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(z_data, x_data, alpha=0.5, label='Datos originales')
    plt.axvline(z_target, color='red', linestyle='--', label=f'z* = {z_target}')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.legend()
    plt.title('Datos originales')
    
    plt.subplot(1, 3, 2)
    plt.scatter(z_data, y_bary, alpha=0.5, label='Baricentro y')
    plt.xlabel('z')
    plt.ylabel('y')
    plt.legend()
    plt.title('Baricentro (y independiente de z)')
    
    plt.subplot(1, 3, 3)
    plt.hist(x_samples, bins=30, alpha=0.7, density=True, label='x|z* simulado')
    plt.axvline(np.mean(x_samples), color='red', linestyle='--', 
                label=f'Media = {np.mean(x_samples):.3f}')
    plt.xlabel('x')
    plt.ylabel('Densidad')
    plt.legend()
    plt.title(f'Distribución condicional ρ(x|z*={z_target})')
    
    plt.tight_layout()
    plt.show()
