import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configuración de semilla para reproducibilidad
np.random.seed(42)

# Parámetros de simulación
T = 252 * 2  # 2 años de datos diarios
dt = 1/252   # paso temporal diario

# Parámetros del proceso de cointegración
# Velocidad de reversión a la media (time-varying)
def theta_t(t):
    """Velocidad de reversión que varía con el tiempo"""
    # Mayor reversión al principio y al final, menor en el medio
    return 0.5 + 0.3 * np.sin(2 * np.pi * t / T)

# Volatilidad dependiente del tiempo
def sigma_t(t):
    """Volatilidad que varía con el tiempo"""
    # Volatilidad cíclica con tendencia
    return 0.15 + 0.1 * np.sin(4 * np.pi * t / T) + 0.05 * (t / T)

# Parámetros del activo 1 (proceso con drift)
mu1 = 0.08  # drift anual
sigma1 = 0.20  # volatilidad

# Parámetro de cointegración (beta)
beta = 1.5

# Inicialización
S1 = np.zeros(T)
S2 = np.zeros(T)
spread = np.zeros(T)

# Valores iniciales
S1[0] = 100
S2[0] = 150
spread[0] = 0

# Simulación del proceso
for t in range(1, T):
    # Ruido aleatorio
    dW1 = np.random.normal(0, np.sqrt(dt))
    dW2 = np.random.normal(0, np.sqrt(dt))
    
    # Activo 1: Geometric Brownian Motion
    S1[t] = S1[t-1] * np.exp((mu1 - 0.5 * sigma1**2) * dt + sigma1 * dW1)
    
    # Spread con reversión a la media (Ornstein-Uhlenbeck)
    theta = theta_t(t)
    sigma = sigma_t(t)
    spread[t] = spread[t-1] - theta * spread[t-1] * dt + sigma * dW2
    
    # Activo 2: depende de S1 y del spread
    S2[t] = beta * S1[t] + spread[t]

# Crear visualizaciones
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 1. Serie temporal de ambos activos
axes[0, 0].plot(S1, label='Activo 1', linewidth=1.5)
axes[0, 0].plot(S2, label='Activo 2', linewidth=1.5)
axes[0, 0].set_title('Series Temporales de los Activos', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Tiempo (días)')
axes[0, 0].set_ylabel('Precio')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Spread (diferencia cointegrada)
axes[0, 1].plot(spread, color='purple', linewidth=1.5)
axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Media')
axes[0, 1].set_title('Spread Cointegrado (S2 - β·S1)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Tiempo (días)')
axes[0, 1].set_ylabel('Spread')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Relación de cointegración (scatter plot)
axes[1, 0].scatter(S1, S2, alpha=0.5, s=10)
axes[1, 0].plot(S1, beta * S1, 'r--', linewidth=2, label=f'Relación: S2 = {beta}·S1')
axes[1, 0].set_title('Relación de Cointegración', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Activo 1')
axes[1, 0].set_ylabel('Activo 2')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Velocidad de reversión en el tiempo
time_array = np.arange(T)
theta_array = [theta_t(t) for t in time_array]
axes[1, 1].plot(theta_array, color='green', linewidth=2)
axes[1, 1].set_title('Velocidad de Reversión θ(t)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Tiempo (días)')
axes[1, 1].set_ylabel('θ(t)')
axes[1, 1].grid(True, alpha=0.3)

# 5. Volatilidad en el tiempo
sigma_array = [sigma_t(t) for t in time_array]
axes[2, 0].plot(sigma_array, color='orange', linewidth=2)
axes[2, 0].set_title('Volatilidad σ(t) del Spread', fontsize=12, fontweight='bold')
axes[2, 0].set_xlabel('Tiempo (días)')
axes[2, 0].set_ylabel('σ(t)')
axes[2, 0].grid(True, alpha=0.3)

# 6. Distribución del spread
axes[2, 1].hist(spread, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
mu_spread, std_spread = spread.mean(), spread.std()
xmin, xmax = axes[2, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu_spread, std_spread)
axes[2, 1].plot(x, p, 'r-', linewidth=2, label='Normal ajustada')
axes[2, 1].set_title('Distribución del Spread', fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel('Spread')
axes[2, 1].set_ylabel('Densidad')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Estadísticas del spread
print("=" * 60)
print("ESTADÍSTICAS DEL SPREAD COINTEGRADO")
print("=" * 60)
print(f"Media: {spread.mean():.4f}")
print(f"Desviación estándar: {spread.std():.4f}")
print(f"Mínimo: {spread.min():.4f}")
print(f"Máximo: {spread.max():.4f}")
print(f"\nVelocidad de reversión θ(t):")
print(f"  - Mínima: {min(theta_array):.4f}")
print(f"  - Máxima: {max(theta_array):.4f}")
print(f"  - Media: {np.mean(theta_array):.4f}")
print(f"\nVolatilidad σ(t):")
print(f"  - Mínima: {min(sigma_array):.4f}")
print(f"  - Máxima: {max(sigma_array):.4f}")
print(f"  - Media: {np.mean(sigma_array):.4f}")
print(f"\nCoeficiente de cointegración (β): {beta}")
print("=" * 60)

# Guardar datos en arrays para uso posterior
print("\nLos datos están disponibles en las variables:")
print("  - S1: Precios del activo 1")
print("  - S2: Precios del activo 2")
print("  - spread: Serie del spread cointegrado")