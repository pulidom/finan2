import numpy as np, os


# --------------------------------------------------
# 1. Función para generar parámetros con cambios de régimen y períodos de no cointegración
# --------------------------------------------------
def generate_regime_parameters_with_breaks(T, regime_length=252, 
                                         break_length=30,
                                         break_prob=0.6,
                                         kappa_range=(0.03, 0.20),
                                         break_kappa_range=(0.001, 0.01),
                                         sigmaX_range=(20*0.01, 20*0.04),
                                         sigmaY_range=(20*0.012, 20*0.05),
                                         rho_range=(-0.9, 0.9),
                                         theta_shift_prob=0.3,
                                         theta_shift_range=(-1.0, 1.0),
                                         seed=None):
    """
    Genera parámetros con cambios de régimen y períodos de no cointegración.
    """
    if seed is not None:
        np.random.seed(seed)

    n_regimes = int(np.ceil(T / regime_length))

    sigma_X_t = np.zeros(T)
    sigma_Y_t = np.zeros(T)
    kappa_t   = np.zeros(T)
    theta_t   = np.zeros(T)
    rho_t     = np.zeros(T)
    regime_labels = np.zeros(T, dtype=int)

    current_time = 0
    
    for r in range(n_regimes):
        # Régimen normal
        regime_end = min(current_time + regime_length, T)
        length = regime_end - current_time
        t_rel = np.arange(length)

        # Parámetros del régimen normal
        k_base = np.random.uniform(*kappa_range)
        k_osc = 0.02 * np.sin(2 * np.pi * t_rel / (length * 2 + 1))
        kappa_t[current_time:regime_end] = np.clip(k_base + k_osc, 0.001, None)

        sX_base = np.random.uniform(*sigmaX_range)
        sY_base = np.random.uniform(*sigmaY_range)
        sX_osc = 0.3 * sX_base * np.sin(2 * np.pi * t_rel / (length + 1))
        sY_osc = 0.3 * sY_base * np.cos(2 * np.pi * t_rel / (length + 1))
        sigma_X_t[current_time:regime_end] = np.clip(sX_base + sX_osc, 1e-6, None)
        sigma_Y_t[current_time:regime_end] = np.clip(sY_base + sY_osc, 1e-6, None)

        if np.random.rand() < theta_shift_prob:
            theta_base = np.random.uniform(*theta_shift_range)
        else:
            theta_base = 0.0
        theta_t[current_time:regime_end] = theta_base

        rho_base = np.random.uniform(*rho_range)
        rho_t[current_time:regime_end] = rho_base
        
        regime_labels[current_time:regime_end] = 0
        current_time = regime_end

        # Período de break
        if current_time < T and np.random.rand() < break_prob and r < n_regimes - 1:
            break_end = min(current_time + break_length, T)
            break_duration = break_end - current_time
            
            kappa_break = np.random.uniform(*break_kappa_range)
            kappa_t[current_time:break_end] = kappa_break
            
            sigma_X_t[current_time:break_end] = sigma_X_t[current_time-1] * np.random.uniform(1.0, 1.5)
            sigma_Y_t[current_time:break_end] = sigma_Y_t[current_time-1] * np.random.uniform(1.0, 1.5)
            
            theta_t[current_time:break_end] = theta_t[current_time-1] + np.random.uniform(-2.0, 2.0)
            
            rho_t[current_time:break_end] = np.random.uniform(*rho_range)
            
            regime_labels[current_time:break_end] = 1
            current_time = break_end

    return sigma_X_t, sigma_Y_t, kappa_t, theta_t, rho_t, regime_labels

# --------------------------------------------------
# 2. Función para simular con períodos de no cointegración
# --------------------------------------------------
def simulate_with_regime_breaks(sigma_X_t, sigma_Y_t, kappa_t, theta_t, rho_t,
                               regime_labels, mu_X=0.0, mu_Y=0.0,
                               X0=100.0, Y0=100.0, dt=1.0):
    """
    Simula dos activos con períodos de cointegración y no cointegración.
    """
    T = len(sigma_X_t)
    X = np.zeros(T)
    Y = np.zeros(T)
    spread_series = np.zeros(T)
    X[0] = X0
    Y[0] = Y0
    spread_series[0] = X0 - Y0

    for t in range(T - 1):
        # Matriz de correlación variable
        rho = rho_t[t]
        cov = np.array([[1, rho],
                       [rho, 1]])
        
        try:
            L = np.linalg.cholesky(cov)
            eps = np.random.randn(2)
            dW = L @ eps
        except np.linalg.LinAlgError:
            dW = np.random.randn(2)

        sX = sigma_X_t[t]
        sY = sigma_Y_t[t]
        k = kappa_t[t]
        th = theta_t[t]

        if regime_labels[t] == 1:  # Período de break
            X[t+1] = X[t] + mu_X * dt + sX * np.sqrt(dt) * dW[0]
            Y[t+1] = Y[t] + mu_Y * dt + sY * np.sqrt(dt) * dW[1]
        else:  # Régimen normal
            X[t+1] = X[t] + mu_X * dt + sX * np.sqrt(dt) * dW[0]
            spread = X[t] - Y[t] - th
            Y[t+1] = Y[t] + mu_Y * dt + k * spread * dt + sY * np.sqrt(dt) * dW[1]

        spread_series[t+1] = X[t+1] - Y[t+1]

    return np.vstack([X, Y])#, spread_series


# --------------------------------------------------
# 1. Función para generar volatilidades, kappa y theta
# --------------------------------------------------
def generate_regime_parameters(T, regime_length=252, 
                               kappa_range=(0.03, 0.20),
                               sigmaX_range=(20*0.01,20* 0.04),
                               sigmaY_range=(20*0.012,20* 0.05),
                               theta_shift_prob=0.3,
                               theta_shift_range=(-1.0, 1.0),
                               seed=None):
    """
    Genera arrays de sigma_X(t), sigma_Y(t), 
             kappa(t)   Velocidad de reversión
             theta(t) valor medio del spread
           con cambios de régimen largos.
    Sin cambio de regimen seria: regime_length=T
    """
    if seed is not None:
        np.random.seed(seed)

    n_regimes = int(np.ceil(T / regime_length))

    sigma_X_t = np.zeros(T)
    sigma_Y_t = np.zeros(T)
    kappa_t   = np.zeros(T)
    theta_t   = np.zeros(T)

    for r in range(n_regimes):
        start = r * regime_length
        end = min((r + 1) * regime_length, T)
        length = end - start
        t_rel = np.arange(length)

        # Kappa base y oscilación
        k_base = np.random.uniform(*kappa_range)
        k_osc = 0.02 * np.sin(2 * np.pi * t_rel / (length * 2 + 1))
        kappa_t[start:end] = np.clip(k_base + k_osc, 0.001, None)

        # Volatilidades base y oscilación
        sX_base = np.random.uniform(*sigmaX_range)
        sY_base = np.random.uniform(*sigmaY_range)
        sX_osc = 0.3 * sX_base * np.sin(2 * np.pi * t_rel / (length + 1))
        sY_osc = 0.3 * sY_base * np.cos(2 * np.pi * t_rel / (length + 1))
        sigma_X_t[start:end] = np.clip(sX_base + sX_osc, 1e-6, None)
        sigma_Y_t[start:end] = np.clip(sY_base + sY_osc, 1e-6, None)

        # Theta base por régimen (opcional)
        if np.random.rand() < theta_shift_prob:
            theta_base = np.random.uniform(*theta_shift_range)
        else:
            theta_base = 0.0
        theta_t[start:end] = theta_base

    return sigma_X_t, sigma_Y_t, kappa_t, theta_t


# --------------------------------------------------
# 2. Función para simular activos cointegrados
# --------------------------------------------------
def simulate_cointegrated_assets(sigma_X_t, sigma_Y_t, kappa_t, theta_t,
                                 mu_X=0.0, mu_Y=0.0, rho=0.8,
                                 X0=100.0, Y0=100.0, dt=1.0):
    """
    Simula dos series cointegradas usando Euler–Maruyama con parámetros dependientes del tiempo.
    """
    T = len(sigma_X_t)
    X = np.zeros(T)
    Y = np.zeros(T)
    X[0] = X0
    Y[0] = Y0

    # correlación entre shocks
    cov = np.array([[1, rho],
                    [rho, 1]])
    L = np.linalg.cholesky(cov)

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

    print(X.shape)
    print(Y.shape)
    
    return np.vstack([X, Y])

def simulate_cointegrated_with_aux(
        
    sigma_X_t, sigma_Y_t, kappa_t, theta_t,
    mu_X=0.0, mu_Y=0.0, rho_XY=0.8,
    X0=100.0, Y0=100.0, 
    # parámetros de Z
    phi_Z=0.95, sigma_Z=0.01, rho_XZ=0.5, Z0=0.0,
    gamma_Z=0.02
):
    """
    Simula X, Y cointegrados + Z correlacionado con X y predictivo.
    """
    T = len(sigma_X_t)
    X = np.zeros(T)
    Y = np.zeros(T)
    Z = np.zeros(T)
    
    X[0], Y[0], Z[0] = X0, Y0, Z0
    spread0 = X0 - Y0
    S = np.zeros(T)
    S[0] = spread0

    for t in range(1, T):
        # shocks correlacionados X-Y
        eps_X = np.random.randn()
        eps_Y = rho_XY * eps_X + np.sqrt(1 - rho_XY**2) * np.random.randn()
        
        # shock de Z correlacionado con X
        eps_Z = rho_XZ * eps_X + np.sqrt(1 - rho_XZ**2) * np.random.randn()
        
        # Spread OU
        S[t] = S[t-1] + kappa_t[t]*(theta_t[t]-S[t-1]) + sigma_Y_t[t]*eps_Y
        
        # Z dynamics
        Z[t] = phi_Z*Z[t-1] + sigma_Z*eps_Z
        
        # X dynamics: correlación + predictibilidad via Z
        X[t] = X[t-1] + mu_X + sigma_X_t[t]*eps_X + gamma_Z*(Z[t-1] - 0.0)
        # Y determined from spread
        Y[t] = X[t] - S[t]
        
    return np.vstack([X, Y,Z])
# ------------------------
# Utils: muestreo multivar t
# ------------------------
def multivariate_t_rvs(mean, cov, df, n):
    """
    Sample n draws from multivariate t with location `mean`, scale `cov` and df.
    Returns array shape (n, dim).
    """
    dim = cov.shape[0]
    # normal draws
    z = np.random.multivariate_normal(np.zeros(dim), cov, size=n)
    # chi-squared draws
    chi2 = np.random.chisquare(df, size=n) / df
    # scale
    t = z / np.sqrt(chi2)[:, None]
    return mean + t

# ------------------------
# Generación parámetros (igual que antes)

# ------------------------
# Simulación con t multivariante
# ------------------------
def simulate_with_heavy_t(
    sigma_X_t, sigma_Y_t, kappa_t, theta_t,
    mu_X=0.0, mu_Y=0.0,
    X0=100.0, Y0=100.0,
    # Z params
    phi_Z=0.98, sigma_Z=0.01, Z0=0.0, gamma_Z=0.02,
    # correlation and heavy-tail
    rho_XY=0.8, rho_XZ=0.6, rho_YZ=0.4, df=5
):
    """
    Simula X,Y cointegrados y Z (aux) usando choques multivariados Student-t.
    df: grados de libertad (menor -> colas más pesadas).
    """
    T = len(sigma_X_t)
    X = np.zeros(T); Y = np.zeros(T); Z = np.zeros(T); S = np.zeros(T)
    X[0], Y[0], Z[0] = X0, Y0, Z0
    S[0] = X0 - Y0

    # construir matriz de correlación para (eps_X, eps_Y, eps_Z)
    R = np.array([[1.0, rho_XY, rho_XZ],
                  [rho_XY, 1.0, rho_YZ],
                  [rho_XZ, rho_YZ, 1.0]])
    # verificar positiva definida (si no, ajustar pequeños valores)
    # escalar por 1 (covariance base); magnitudes efectivas vendrán por sigma_* y sigma_Z
    mean = np.zeros(3)

    for t in range(T - 1):
        # sample multivariate t (una muestra)
        eps = multivariate_t_rvs(mean, R, df, 1)[0]  # shape (3,)
        eps_X, eps_Y, eps_Z = eps  # heavy-tailed shocks

        # escalar shocks por volatilidades locales
        shock_X = sigma_X_t[t] * eps_X
        shock_Y = sigma_Y_t[t] * eps_Y
        shock_Z = sigma_Z * eps_Z

        # actualizar Z (AR(1) con choque heavy-tailed correlacionado)
        Z[t+1] = phi_Z * Z[t] + shock_Z

        # actualizar spread S como OU discretizado (usamos sigma_Y para ruido del spread)
        S[t+1] = S[t] + kappa_t[t] * (theta_t[t] - S[t]) + shock_Y

        # X dynamics: además de su choque heavy-tailed, añade predictibilidad desde Z
        X[t+1] = X[t] + mu_X + shock_X + gamma_Z * (Z[t] - 0.0)

        # Y definido por X - spread
        Y[t+1] = X[t+1] - S[t+1]

    return  np.vstack([X, Y, Z])

def load_sts(nt=5*252,lopt=0,regime_length=None):
    '''  lopt=0 two cointegrated assets
         lopt=1 a third covariate variable
         lopt=2 a third covariate variable and non-gaussian heavy tail stats
    '''
    if regime_length==None: regime_length= nt
    sigma_X_t, sigma_Y_t, kappa_t, theta_t = generate_regime_parameters(nt, regime_length, seed=42)
    if lopt==0:
        ts = simulate_cointegrated_assets(sigma_X_t, sigma_Y_t, kappa_t, theta_t)
    elif lopt==1:
        ts = simulate_cointegrated_with_aux(sigma_X_t, sigma_Y_t, kappa_t, theta_t)
    elif lopt==2:
        ts = simulate_with_heavy_t(sigma_X_t, sigma_Y_t, kappa_t, theta_t)

    elif lopt==3:
        
        sigma_X_t, sigma_Y_t, kappa_t, theta_t, rho_t, regime_labels = generate_regime_parameters_with_breaks(
            nt, regime_length=regime_length, break_length=30, seed=42
        )
    
    # Simular
        ts = simulate_with_regime_breaks(
        sigma_X_t, sigma_Y_t, kappa_t, theta_t, rho_t, regime_labels
        )
        
    return ts

# --------------------------------------------------
# 3. Ejemplo de uso
# --------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    T = 5 * 252
    sigma_X_t = np.full(T, 0.02)
    sigma_Y_t = np.full(T, 0.025)
    kappa_t   = np.full(T, 0.1)
    theta_t   = np.zeros(T)

    X, Y, Z = simulate_cointegrated_with_aux(
        sigma_X_t, sigma_Y_t, kappa_t, theta_t,
        rho_XY=0.8, rho_XZ=0.6, gamma_Z=0.03,
        phi_Z=0.98, sigma_Z=0.01
    )

    print(np.array([X,Y,Z]).shape)
    quit()
    
    t_axis = np.arange(T)

    plt.figure(figsize=(12,5))
    plt.plot(t_axis, X, label='X')
    plt.plot(t_axis, Y, label='Y')
    plt.plot(t_axis, Z, label='Z')
    plt.title('Series simuladas con variable auxiliar Z')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12,3))
    plt.plot(t_axis, S)
    plt.title('Spread X - Y')
    plt.grid()

    plt.figure(figsize=(12,3))
    plt.scatter(Z[:-1], np.diff(X), s=5, alpha=0.5)
    plt.xlabel('Z_t')
    plt.ylabel('ΔX_{t+1}')
    plt.title('Relación predictiva entre Z y ΔX')
    plt.grid()
    plt.show()
