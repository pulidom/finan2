import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MeanRevertingAssets:
    """Generate synthetic mean-reverting assets using Ornstein-Uhlenbeck process"""
    
    def __init__(self, n_periods=1000, dt=1/252):
        self.n_periods = n_periods
        self.dt = dt
        
    def generate_ou_process(self, theta=0.1, mu=0, sigma=0.2, x0=0):
        """
        Generate Ornstein-Uhlenbeck process
        dx = theta * (mu - x) * dt + sigma * dW
        
        Parameters:
        - theta: speed of mean reversion
        - mu: long-term mean
        - sigma: volatility
        - x0: initial value
        """
        x = np.zeros(self.n_periods)
        x[0] = x0
        
        for t in range(1, self.n_periods):
            dx = theta * (mu - x[t-1]) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal()
            x[t] = x[t-1] + dx
            
        return x
    
    def generate_cointegrated_pair(self, theta1=0.15, theta2=0.12, mu1=100, mu2=50, 
                                  sigma1=0.25, sigma2=0.20, correlation=0.7, beta=1.5):
        """
        Generate two cointegrated mean-reverting assets
        Asset2 = beta * Asset1 + error (where error is mean-reverting)
        """
        # Generate correlated innovations
        innovations = np.random.multivariate_normal(
            [0, 0], 
            [[1, correlation], [correlation, 1]], 
            self.n_periods
        )
        
        # Generate first asset (mean-reverting)
        asset1 = np.zeros(self.n_periods)
        asset1[0] = mu1
        
        for t in range(1, self.n_periods):
            dx1 = theta1 * (mu1 - asset1[t-1]) * self.dt + sigma1 * np.sqrt(self.dt) * innovations[t, 0]
            asset1[t] = asset1[t-1] + dx1
        
        # Generate second asset with cointegration relationship
        asset2 = np.zeros(self.n_periods)
        spread = self.generate_ou_process(theta=0.2, mu=0, sigma=0.1, x0=0)  # Mean-reverting spread
        
        asset2 = beta * asset1 + mu2 - beta * mu1 + spread
        
        return asset1, asset2

def test_linear_models(X, y, test_size=0.2):
    """Test different linear modeling approaches"""
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features for regularized methods
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
    
    # Define models to test
    models = {
        'OLS': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Lasso (α=1.0)': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Huber': HuberRegressor(),
    }
    
    results = {}
    
    for name, model in models.items():
        # Use scaled data for regularized methods, original for OLS
        if name == 'OLS':
            model.fit(X_train.reshape(-1, 1), y_train)
            y_pred = model.predict(X_test.reshape(-1, 1))
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get coefficient (adjust for scaling if needed)
        if name == 'OLS':
            coef = model.coef_[0]
        else:
            coef = model.coef_[0] / scaler.scale_[0]
        
        intercept = model.intercept_
        
        results[name] = {
            'model': model,
            'coefficient': coef,
            'intercept': intercept,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    return results, X_train, X_test, y_train, y_test

def analyze_cointegration(asset1, asset2):
    """Analyze cointegration relationship between assets"""
    from scipy.stats import pearsonr
    
    # Calculate correlation
    correlation, p_value = pearsonr(asset1, asset2)
    
    # Fit simple linear regression to get residuals
    X = asset1.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, asset2)
    residuals = asset2 - model.predict(X)
    
    # Test residuals for stationarity (simplified ADF test approximation)
    # In practice, you'd use statsmodels for proper ADF test
    residuals_diff = np.diff(residuals)
    residuals_lag = residuals[:-1]
    
    # Simple regression: diff = alpha * lag + error
    slope, intercept, r_val, p_val, std_err = stats.linregress(residuals_lag, residuals_diff)
    
    return {
        'correlation': correlation,
        'correlation_pvalue': p_value,
        'beta': model.coef_[0],
        'alpha': model.intercept_,
        'residuals': residuals,
        'adf_stat_approx': slope,  # Approximation of ADF stat
        
