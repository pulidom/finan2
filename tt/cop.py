import numpy as np
from scipy import stats as ss
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, multivariate_normal

def empirical_cdf(data):
    ranks = ss.rankdata(data)
    return ranks / (len(data) + 1)

def to_uniform_margins(data):
    ecdf = ECDF(data)
    return ecdf(data)

class Gaussian:
    def __init__(self):
        self.rho = None  # correlación
        self.corr_matrix = None
        self.historical_dependencies = []  # Para trackear dependencias históricas

    def fit(self, data):
        """
        data: np.ndarray of shape (n_samples, 2), assumed to be uniform marginals (in [0,1])
        """
        if data.shape[1] != 2:
            raise ValueError("Only bivariate Gaussian copula is supported.")

        # Transform to normal marginals
        norm_data = norm.ppf(data)

        # Estimate correlation matrix
        self.corr_matrix = np.corrcoef(norm_data.T)
        self.rho = self.corr_matrix[0, 1]
        
        # Guardar dependencia histórica
        self.historical_dependencies.append(self.rho)

    def sample(self, n_samples):
        if self.corr_matrix is None:
            raise ValueError("Model must be fit before sampling.")

        mean = np.zeros(2)
        samples = multivariate_normal.rvs(mean=mean, cov=self.corr_matrix, size=n_samples)

        # Transform back to uniform
        return norm.cdf(samples)

    def log_likelihood(self, data):
        """
        Computes the log-likelihood of the copula for uniform marginals.
        """
        if self.corr_matrix is None:
            raise ValueError("Model must be fit before computing log-likelihood.")

        norm_data = norm.ppf(data)
        mvn_logpdf = multivariate_normal.logpdf(norm_data, mean=np.zeros(2), cov=self.corr_matrix)
        ind_logpdf = np.sum(norm.logpdf(norm_data), axis=1)
        return mvn_logpdf - ind_logpdf
    
    def compute_dependence_zscore(self, new_data):
        """
        Calcula el z-score de dependencia entre dos activos
        
        Parameters:
        new_data: np.ndarray of shape (n_samples, 2) - nuevos datos para comparar
        
        Returns:
        z_score: qué tan extrema es la dependencia actual vs histórica
        current_dependence: medida de dependencia actual
        """
        if self.corr_matrix is None:
            raise ValueError("Model must be fit before computing z-score.")
        
        # Transformar nuevos datos a normales
        norm_new_data = norm.ppf(new_data)
        
        # Calcular dependencia actual (correlación)
        current_corr_matrix = np.corrcoef(norm_new_data.T)
        current_dependence = current_corr_matrix[0, 1]
        
        # Si tenemos suficiente historia, calcular z-score
        if len(self.historical_dependencies) > 1:
            historical_mean = np.mean(self.historical_dependencies)
            historical_std = np.std(self.historical_dependencies)
            
            if historical_std > 0:
                z_score = (current_dependence - historical_mean) / historical_std
            else:
                z_score = 0  # No hay variabilidad histórica
        else:
            z_score = 0  # No hay suficiente historia
        
        return z_score#, current_dependence

# FUNCIÓN ESPECÍFICA PARA ACTIVOS
def compute_assets_copula_zscore(asset1_returns, asset2_returns, window_size=252):
    """
    Calcula z-score de dependencia entre dos activos usando cópula gaussiana
    
    Parameters:
    asset1_returns, asset2_returns: returns de los activos
    window_size: tamaño de ventana para cálculo histórico
    
    Returns:
    z_score: z-score de la dependencia actual
    current_corr: correlación actual
    """
    # Convertir returns a uniform marginals
    uniform_data1 = to_uniform_margins(asset1_returns)
    uniform_data2 = to_uniform_margins(asset2_returns)
    
    # Combinar datos
    uniform_data = np.column_stack([uniform_data1, uniform_data2])
    
    # Crear y ajustar cópula
    copula = Gaussian()
    
    # Usar ventana deslizante para historia
    if len(uniform_data) > window_size:
        historical_data = uniform_data[:-window_size]
        current_data = uniform_data[-window_size:]
        
        # Ajustar con datos históricos
        copula.fit(historical_data)
        
        # Calcular z-score para datos actuales
        z_score, current_corr = copula.compute_dependence_zscore(current_data)
    else:
        # Datos insuficientes, usar todos
        copula.fit(uniform_data)
        z_score, current_corr = 0, copula.rho
    
    return z_score, current_corr

# EJEMPLO DE USO
if __name__ == "__main__":
    # Simular datos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    
    # Simular returns de dos activos correlacionados
    returns1 = np.random.normal(0, 0.02, n_samples)
    returns2 = 0.7 * returns1 + np.random.normal(0, 0.015, n_samples)
    
    # Calcular z-score
    z_score, current_corr = compute_assets_copula_zscore(returns1, returns2)
    
    print(f"Correlación actual: {current_corr:.4f}")
    print(f"Z-score de dependencia: {z_score:.4f}")
    
    # Interpretación
    if abs(z_score) > 2:
        print("ALERTA: Dependencia significativamente anormal")
    elif abs(z_score) > 1:
        print("Advertencia: Dependencia moderadamente anormal")
    else:
        print("Dependencia dentro de rangos normales")

















def gaussian_copula_log_likelihood(theta, u, v):
    rho = np.tanh(theta[0])  # Constrain rho to [-1, 1]
    cov = np.array([[1.0, rho], [rho, 1.0]])
    inv_cov = np.linalg.inv(cov)
    log_det = np.log(np.linalg.det(cov))
    
    z_u = norm.ppf(u)
    z_v = norm.ppf(v)
    z = np.column_stack([z_u, z_v])  # Shape (n_samples, 2)
    
    # Compute z^T * inv_cov * z for each observation
    quad_form = np.sum(z.dot(inv_cov) * z, axis=1)  # Shape (n_samples,)
    
    # Sum over all observations
    nll = -0.5 * np.sum(quad_form + log_det)
    return -nll  # Minimize negative log-likelihood

# Step 4: Generate trading signals using copula probabilities
def generate_signals(u, v, rho, threshold=0.95):
    z_u = norm.ppf(u)
    z_v = norm.ppf(v)
    cond_mean = rho * z_u
    cond_std = np.sqrt(1 - rho**2)
    prob_v_given_u = norm.cdf(z_v, loc=cond_mean, scale=cond_std)
    
    # Long when prob < 0.05, short when prob > 0.95
    signals = np.zeros(len(u))
    signals[prob_v_given_u < (1 - threshold)] = 1  # Y is too low relative to X
    signals[prob_v_given_u > threshold] = -1       # Y is too high relative to X
    return signals

        


# Rank-transform to uniform margins (PIT)
u = np.argsort(np.argsort(x)) / (len(x) + 1)
v = np.argsort(np.argsort(y)) / (len(y) + 1)

# Fit Gaussian copula
initial_theta = 0.5
res = minimize(gaussian_copula_log_likelihood, initial_theta, args=(u, v), method='BFGS')
rho = np.tanh(res.x[0])  # Estimated correlation
print(f"Estimated Gaussian Copula Rho: {rho:.4f}")
signals = generate_signals(u, v, rho, threshold=0.95)
