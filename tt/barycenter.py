import numpy as np    
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.cluster import KMeans
class MongeOTBP:
    """
    Monge Optimal Transport Barycenter Problem
    basada en el paper de Lipnick et al.
    """
    def __init__(self, lambda_param=None, alpha=0.0025, delta=0.1, 
                 nu=0.2, theta=1.1, kappa=0.5, tau=0.5, K_max=5):
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.delta = delta
        self.nu = nu
        self.theta = theta
        self.kappa = kappa
        self.tau = tau
        self.K_max = K_max
        self.history = []
    
    def _gaussian_kernel(self, x, centers, bandwidth):
        """Funcion kernel gaussiana"""
        x = np.atleast_2d(x)
        centers = np.atleast_2d(centers)
        
        if x.shape[1] != centers.shape[1]:
            x = x.T if x.shape[0] == centers.shape[1] else x
            
        dist_sq = np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-dist_sq / (2 * bandwidth ** 2))
    
    def _create_gaussian_spaces(self, data, n_centers=20, bandwidth=None):
        """Crear espacios funcionales F y G usando kernels gaussianos"""
        n_samples = len(data)
        
        if bandwidth is None:
            bandwidth = np.std(data) / 2
        
        # Usar k-means para encontrar centros
        if len(data.shape) == 1:
            data_reshaped = data.reshape(-1, 1)
        else:
            data_reshaped = data
            
        n_centers = min(n_centers, n_samples)
        
        if n_centers == 1:
            centers = np.mean(data_reshaped, axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
            centers = kmeans.fit(data_reshaped).cluster_centers_

        # realmente propongo el numero de centros o a partir de cada
        # particula/muestra genero un centro?

        # Crear matriz de funciones
        func_matrix = self._gaussian_kernel(data_reshaped, centers, bandwidth)
        #func_matrix = func_matrix - np.mean(func_matrix, axis=0)# centro deepseek
      
        return func_matrix, centers, bandwidth
    
    def _orthogonalize_functions(self, func_matrix):
        """Ortogonalizar matriz de funciones usando SVD"""
        
        U, s, Vt = svd(func_matrix, full_matrices=False)
        
        # Determinar rango efectivo
        s_squared = s ** 2
        total_energy = np.sum(s_squared)
        cumsum_energy = np.cumsum(s_squared)
        
        # Mantener componentes que explican al menos 99% de la energia
        threshold = 0.99 * total_energy
        effective_rank = np.searchsorted(cumsum_energy, threshold) + 1
        effective_rank = min(effective_rank, len(s))
        
        # Crear matriz ortogonal
        Q = U[:, :effective_rank] #@ np.diag(Sz)???? ## deepseek
        B = (Vt[:effective_rank, :] / s[:effective_rank, np.newaxis]).T
        
        return Q, B
    
    def initialize_F_space(z):
        """ Espacio F para z (parametros) """
        
        F, centers, bandwidth = _create_gaussian_spaces(z)
        Qz,Bz = _orthogonalize_functions(F)

        return Qz,Bz,centers_z
    
     def initialize_G_space(y):
      """Espacio G para y (parámetros transformados) """
      
        G = np.column_stack([np.ones_like(y), y, y**2, np.sin(y), np.cos(y)])
        G_centered = G - np.mean(G, axis=0)
        Qy,By = _orthogonalize_functions(Gcentered)

        return Qy,By,0
