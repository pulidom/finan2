import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parámetros de la simulación
# ----------------------------
T = 5 * 252        # 5 años con 252 días hábiles
dt = 1.0           # paso temporal diario
rho = 0.8          # correlación entre shocks

# Medias (pueden ser cero para simplificar)
mu_X = 0.0
mu_Y = 0.0

# Inicialización de arrays
X = np.zeros(T)
Y = np.zeros(T)

# Valores iniciales
X[0] = 100.0
Y[0] = 100.0

# Generar choques correlacionados
# Matriz de covarianza
cov = np.array([[1, rho],
                [rho, 1]])
L = np.linalg.cholesky(cov)   # descomposición de Cholesky

# ----------------------------
# Funciones dependientes del tiempo
# ----------------------------
def sigma_X(t):
    # Volatilidad estacional sinusoidal anual
    return 0.02 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 250.0))

def sigma_Y(t):
    return 0.025 * (1.0 + 0.5 * np.cos(2 * np.pi * t / 250.0))

def kappa(t):
    # Velocidad de reversión que oscila más rápido
    return 0.1 + 0.05 * np.cos(2 * np.pi * t / 100.0)

def theta(t):
    # Spread de equilibrio, cero aquí
    return 0.0

# ----------------------------
# Simulación con Euler-Maruyama
# ----------------------------
for t in range(T - 1):
    # generar shocks correlacionados
    eps = np.random.randn(2)
    dW = L @ eps

    sX = sigma_X(t)
    sY = sigma_Y(t)
    k = kappa(t)
    th = theta(t)

    # Evolución de X (proceso libre)
    X[t+1] = X[t] + mu_X * dt + sX * np.sqrt(dt) * dW[0]

    # Evolución de Y (con reversión hacia X - theta)
    spread = X[t] - Y[t] - th
    Y[t+1] = Y[t] + mu_Y * dt + k * spread * dt + sY * np.sqrt(dt) * dW[1]

# ----------------------------
# Visualización
# ----------------------------
t_axis = np.arange(T)

plt.figure(figsize=(12,5))
plt.plot(t_axis, X, label='X_t', lw=1.2)
plt.plot(t_axis, Y, label='Y_t', lw=1.2)
plt.title('Dos series cointegradas con volatilidad y reversión dependientes del tiempo')
plt.xlabel('t')
plt.ylabel('Precio')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el spread
spread_series = X - Y
plt.figure(figsize=(12,4))
plt.plot(t_axis, spread_series, color='purple', lw=1)
plt.title('Spread X_t - Y_t')
plt.xlabel('t')
plt.ylabel('Spread')
plt.grid(True)
plt.show()
