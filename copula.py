import numpy as np
from scipy.stats import norm
#from scipy import stats as ss
#from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, multivariate_normal

def to_uniform_margins(data):
    #u = np.argsort(np.argsort(x)) / (len(x) + 1)
    #v = np.argsort(np.argsort(y)) / (len(y) + 1)
    ecdf = ECDF(data)
    return ecdf(data)

class GaussianCopula:
    def __init__(self):
        self.rho = None
        self.corr_matrix=None

    def to_uniform(self,x,y):
        ''' transform to uniform '''
#        ecdf_x = ECDF(x)
#        ecdf_y = ECDF(x)
#        return ecdf_x(x),ecdf_y(y)
        # alternativa
        u = np.argsort(np.argsort(x)) / (len(x) + 1)
        v = np.argsort(np.argsort(y)) / (len(y) + 1)
        return u,v
        
    def fit(self, u,v):
        """
        data: np.ndarray of shape (n_samples, 2), assumed to be uniform marginals (in [0,1])
        """
        u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
        v_clipped = np.clip(v, 1e-10, 1 - 1e-10)
    
        data = np.column_stack([u_clipped, v_clipped])        
        # Transform to normal marginals
        norm_data = norm.ppf(data)

        # Estimate correlation matrix
        self.corr_matrix = np.corrcoef(norm_data.T)
        self.rho = self.corr_matrix[0, 1]
        # Alternativa robusta para valores outliers
        #         tau, _ = kendalltau(u, v)
        #        self.rho = np.sin(np.pi * tau / 2)
       
    def sample(self, n_samples):
        if self.corr_matrix is None:
            raise ValueError("Model must be fit before sampling.")

        mean = np.zeros(2)
        samples = multivariate_normal.rvs(mean=mean, cov=self.corr_matrix, size=n_samples)

        # Transform back to uniform
        return norm.cdf(samples)

    def log_likelihood(self, u,v):
        """
        Computes the log-likelihood of the copula for uniform marginals.
        """
        if self.corr_matrix is None:
            raise ValueError("Model must be fit before computing log-likelihood.")

        data=np.column_stack([u, v])
        norm_data = norm.ppf(data)
        mvn_logpdf = multivariate_normal.logpdf(norm_data, mean=np.zeros(2), cov=self.corr_matrix)
        ind_logpdf = np.sum(norm.logpdf(norm_data), axis=1)
        return mvn_logpdf - ind_logpdf
    
    def conditional_probability(self, u, v):
        """P(V <= v | U = u)"""

        z_u = norm.ppf(np.clip(u, 1e-10, 1-1e-10))
        z_v = norm.ppf(np.clip(v, 1e-10, 1-1e-10))
        
        cond_mean = self.rho * z_u
        cond_std = np.sqrt(1 - self.rho**2)
        
        return norm.cdf(z_v, loc=cond_mean, scale=cond_std)
    
    def conditional_zscore(self, u, v):
        """
        Transforma P(V ≤ v | U = u) a Z-score estándar ~ N(0,1)       
        """
        prob_cond = self.conditional_probability(u, v)
        
        # Transformar a Z-score: prob → z
        z_score = norm.ppf(prob_cond)
        
        return z_score
#        z_scores = self.conditional_zscore(u, v)
    
    def generate_trading_signals(self,z_scores, z_threshold=2.0):
        """
        Genera señales usando Z-scores estandarizados
        """
        
        signals = np.zeros(len(z_scores))
        signals[z_scores < -z_threshold] = 1    # COMPRA: V muy bajo vs esperado dado U
        signals[z_scores > z_threshold] = -1    # VENTA: V muy alto vs esperado dado U
        
        return signals

def copula_zscore(x,y):
    copula = GaussianCopula()
    u,v = copula.to_uniform(x,y)
    copula.fit(u, v)
    z_scores = copula.conditional_zscore(u, v)
    return z_scores
