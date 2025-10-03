import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parámetros de la simulación
# ----------------------------
np.random.seed(42)   # opcional: fija semilla para reproducibilidad

T = 5 * 252        # 5 años con 252 días hábiles
dt = 1.0
rho = 0.8

mu_X = 0.0
mu_Y = 0.0

X = np.zeros(T)
Y = np.zeros(T)
X[0] = 100.0
Y[0] = 100.0

cov = np.array([[1, rho],
                [rho, 1]])
L = np.linalg.cholesky(cov)

# ----------------------------
# Configuración de regímenes largos
# ----------------------------
regime_length = 252          # duración de cada régimen (en días). Cambialo si querés 504, 756, etc.
n_regimes = int(np.ceil(T / regime_length))

# Rango para muestrear parámetros por régimen (ajustar según necesidad)
kappa_min, kappa_max = 0.03, 0.20      # kappa base por régimen
sigmaX_base_min, sigmaX_base_max = 0.01, 0.04
sigmaY_base_min, sigmaY_base_max = 0.012, 0.05

# Generar arrays temporales de parámetros
sigma_X_t = np.zeros(T)
sigma_Y_t = np.zeros(T)
kappa_t   = np.zeros(T)
theta_t   = np.zeros(T)   # lo dejamos 0 pero es fácil modificarlo por régimen

for r in range(n_regimes):
    start = r * regime_length
    end = min((r + 1) * regime_length, T)
    length = end - start

    # Parametros base aleatorios por régimen
    k_base = kappa_min + (kappa_max - kappa_min) * np.random.rand()
    sX_base = sigmaX_base_min + (sigmaX_base_max - sigmaX_base_min) * np.random.rand()
    sY_base = sigmaY_base_min + (sigmaY_base_max - sigmaY_base_min) * np.random.rand()

    # Pequeña oscilación intra-regimen (no necesaria, pero da variabilidad)
    t_rel = np.arange(length)
    intra_osc_k = 0.02 * np.sin(2 * np.pi * t_rel / (length * 2 + 1))   # muy lenta dentro del régimen
    intra_osc_sX = 0.3 * sX_base * np.sin(2 * np.pi * t_rel / (length + 1))
    intra_osc_sY = 0.3 * sY_base * np.cos(2 * np.pi * t_rel / (length + 1))

    kappa_t[start:end] = np.clip(k_base + intra_osc_k, 0.001, None)   # evitar kappa negativa
    sigma_X_t[start:end] = np.clip(sX_base + intra_osc_sX, 1e-6, None)
    sigma_Y_t[start:end] = np.clip(sY_base + intra_osc_sY, 1e-6, None)

# ----------------------------
# Simulación con Euler-Maruyama
# ----------------------------
for t in range(T - 1):
    eps = np.random.randn(2)
    dW = L @ eps

    sX = sigma_X_t[t]
    sY = sigma_Y_t[t]
    k  = kappa_t[t]
    th = theta_t[t]

    X[t+1] = X[t] + mu_X * dt + sX * np.sqrt(dt) * dW[0]
    spread = X[t] - Y[t] - th
    Y[t+1] = Y[t] + mu_Y * dt + k * spread * dt + sY * np.sqrt(dt) * dW[1]

# ----------------------------
# Plots
# ----------------------------
t_axis = np.arange(T)

plt.figure(figsize=(12,5))
plt.plot(t_axis, X, label='X_t', lw=1.2)
plt.plot(t_axis, Y, label='Y_t', lw=1.2)
plt.title('Dos series cointegradas con cambios de régimen largos')
plt.xlabel('t (días)')
plt.ylabel('Precio')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12,3))
plt.plot(t_axis, X - Y, lw=1)
plt.title('Spread X_t - Y_t')
plt.xlabel('t (días)')
plt.grid(True)

plt.figure(figsize=(12,3))
plt.plot(t_axis, kappa_t, lw=1)
plt.title('kappa(t) por día (regímenes largos)')
plt.xlabel('t (días)')
plt.grid(True)

plt.figure(figsize=(12,3))
plt.plot(t_axis, sigma_X_t, label='sigma_X(t)', lw=1)
plt.plot(t_axis, sigma_Y_t, label='sigma_Y(t)', lw=1)
plt.title('Volatilidades por día (regímenes largos)')
plt.legend()
plt.grid(True)

plt.show()
