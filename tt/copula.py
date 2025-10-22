import numpy as np
from scipy.stats import norm, t, kendalltau, rankdata

class GaussianCopula:
    def __init__(self):
        self.rho = None

    def fit(self, u, v):
        tau, _ = kendalltau(u, v)
        self.rho = np.sin(np.pi * tau / 2)
        return self

    def sample(self, n_samples):
        # Generar muestras de una distribución normal multivariada
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        samples = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
        # Transformar a uniformes usando la CDF normal
        u = norm.cdf(samples[:, 0])
        v = norm.cdf(samples[:, 1])
        return np.column_stack([u, v])
    
    def log_likelihood(self, u, v):
        z_u = norm.ppf(u)
        z_v = norm.ppf(v)
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        inv_cov = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov))
        z = np.column_stack([z_u, z_v])
        quad_form = np.sum(z @ inv_cov * z, axis=1)
        return -0.5 * (quad_form + log_det).sum()

class StudentTCopula:
    def __init__(self):
        self.rho = None
        self.df = None

    def fit(self, u, v):
        tau, _ = kendalltau(u, v)
        self.rho = np.sin(np.pi * tau / 2)
        self.df = 5.0  # Valor inicial (puede optimizarse)
        return self

    def sample(self, n_samples):
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        samples = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
        # Aplicar la transformación t-Student
        u = t.cdf(samples[:, 0], df=self.df)
        v = t.cdf(samples[:, 1], df=self.df)
        return np.column_stack([u, v])
    
    def log_likelihood(self, u, v):
        z_u = t.ppf(u, df=self.df)
        z_v = t.ppf(v, df=self.df)
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        inv_cov = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov))
        z = np.column_stack([z_u, z_v])
        quad_form = np.sum(z @ inv_cov * z, axis=1)
        log_const = np.log(1 + quad_form / self.df) * (- (self.df + 2) / 2)
        return log_const.sum() - 0.5 * log_det

class ClaytonCopula:
    def __init__(self):
        self.theta = None

    def fit(self, u, v):
        # Estimación de theta mediante tau de Kendall
        tau, _ = kendalltau(u, v)
        self.theta = 2 * tau / (1 - tau) if tau != 1 else 10  # Evitar división por cero
        return self

    def sample(self, n_samples):
        # Algoritmo para cópulas arquimedianas
        u = np.random.uniform(0, 1, n_samples)
        w = np.random.uniform(0, 1, n_samples)
        # Transformación condicional inversa
        v = (1 + w**(-self.theta / (1 + self.theta)) * u**(-self.theta) - 1)**(-1/self.theta)
        return np.column_stack([u, v])
    
    def log_likelihood(self, u, v):
        if self.theta <= 0:
            return -np.inf  # theta debe ser > 0
        cdf = (u ** (-self.theta) + v ** (-self.theta) - 1) ** (-1 / self.theta)
        pdf = (1 + self.theta) * (u * v) ** (-self.theta - 1) * cdf ** (self.theta + 2)
        return np.log(pdf).sum()

class GumbelCopula:
    def __init__(self):
        self.theta = None

    def fit(self, u, v):
        tau, _ = kendalltau(u, v)
        self.theta = 1 / (1 - tau) if tau != 1 else 10  # Evitar división por cero
        return self

    def sample(self, n_samples):
        u = np.random.uniform(0, 1, n_samples)
        w = np.random.uniform(0, 1, n_samples)
        # Usar la función de distribución condicional
        v = np.exp(-((-np.log(u))**self.theta + (-np.log(w))**(self.theta))**(1/self.theta))
        return np.column_stack([u, v])
                   
    def log_likelihood(self, u, v):
        if self.theta < 1:
            return -np.inf  # theta debe ser >= 1
        u_theta = (-np.log(u)) ** self.theta
        v_theta = (-np.log(v)) ** self.theta
        cdf = np.exp(-(u_theta + v_theta) ** (1 / self.theta))
        pdf = cdf * (u_theta + v_theta) ** (-2 + 2 / self.theta) * (np.log(u) * np.log(v)) ** (self.theta - 1)
        pdf *= (u_theta + v_theta) ** (1 / self.theta) + self.theta - 1
        return np.log(pdf).sum()

class FrankCopula:
    def __init__(self):
        self.theta = None

    def fit(self, u, v):
        # Estimación inicial de theta (puede optimizarse)
        self.theta = 5.0
        return self
    
    def sample(self, n_samples):
        u = np.random.uniform(0, 1, n_samples)
        w = np.random.uniform(0, 1, n_samples)
        # Transformación para Frank
        v = -1/self.theta * np.log(1 + w * (np.exp(-self.theta) - 1) / (1 + (w - 1) * np.exp(-self.theta * u)))
        return np.column_stack([u, v])
                   
    def log_likelihood(self, u, v):
        if self.theta == 0:
            return -np.inf
        term = (np.exp(-self.theta * u) - 1) * (np.exp(-self.theta * v) - 1)
        cdf = -1 / self.theta * np.log(1 + term / (np.exp(-self.theta) - 1))
        pdf = self.theta * (np.exp(-self.theta * (u + v)) * (np.exp(-self.theta) - 1)) / \
              ((np.exp(-self.theta * u) + np.exp(-self.theta * v) - np.exp(-self.theta * (u + v)) - (np.exp(-self.theta) - 1)) ** 2)
        return np.log(pdf).sum()
