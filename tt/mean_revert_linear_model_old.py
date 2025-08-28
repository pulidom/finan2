import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.stats import norm, rankdata
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MeanRevertingAssets:
    """Generate synthetic mean-reverting assets using Ornstein-Uhlenbeck process"""
    
    def __init__(self, n_periods=1000, dt=1/252):
        self.n_periods = n_periods
        self.dt = dt
        
    def ornstein_uhlenbeck(self, theta, mu, sigma, x0):
        """
        Generate Ornstein-Uhlenbeck process
        dx = theta * (mu - x) * dt + sigma * dW
        """
        x = np.zeros(self.n_periods)
        x[0] = x0
        
        for i in range(1, self.n_periods):
            dW = np.random.normal(0, np.sqrt(self.dt))
            dx = theta * (mu - x[i-1]) * self.dt + sigma * dW
            x[i] = x[i-1] + dx
            
        return x
    
    def generate_assets(self):
        """Generate two correlated mean-reverting assets"""
        # Asset 1 parameters
        theta1, mu1, sigma1, x0_1 = 0.5, 100, 10, 95
        
        # Asset 2 parameters  
        theta2, mu2, sigma2, x0_2 = 0.3, 50, 8, 48
        
        # Generate independent processes
        asset1 = self.ornstein_uhlenbeck(theta1, mu1, sigma1, x0_1)
        asset2_independent = self.ornstein_uhlenbeck(theta2, mu2, sigma2, x0_2)
        
        # Add correlation by mixing with asset1
        correlation = 0.7
        noise = np.random.normal(0, 5, self.n_periods)
        asset2 = correlation * asset1 * 0.4 + (1-correlation) * asset2_independent + noise
        
        return asset1, asset2

class GaussianCopula:
    """Implementation of Gaussian Copula for modeling dependence structure"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.marginal_cdfs = None
        
    def fit_marginals(self, X):
        """Fit marginal distributions using empirical CDF"""
        n_samples, n_features = X.shape
        self.marginal_cdfs = []
        
        for i in range(n_features):
            # Use empirical CDF
            sorted_data = np.sort(X[:, i])
            def empirical_cdf(x, data=sorted_data):
                return np.searchsorted(data, x, side='right') / len(data)
            
            self.marginal_cdfs.append({
                'data': sorted_data,
                'cdf': empirical_cdf
            })
    
    def transform_to_uniform(self, X):
        """Transform data to uniform marginals using fitted CDFs"""
        n_samples, n_features = X.shape
        U = np.zeros_like(X)
        
        for i in range(n_features):
            # Transform to uniform [0,1] using empirical CDF
            ranks = rankdata(X[:, i], method='average')
            U[:, i] = ranks / (len(ranks) + 1)  # Avoid 0 and 1
            
        return U
    
    def transform_to_normal(self, U):
        """Transform uniform marginals to standard normal"""
        # Avoid extreme values that cause inf in inverse normal
        U_clipped = np.clip(U, 1e-6, 1-1e-6)
        return norm.ppf(U_clipped)
    
    def fit_copula(self, X):
        """Fit Gaussian copula to the data"""
        # Transform to uniform marginals
        U = self.transform_to_uniform(X)
        
        # Transform to standard normal
        Z = self.transform_to_normal(U)
        
        # Estimate correlation matrix
        self.correlation_matrix = np.corrcoef(Z.T)
        
        return self.correlation_matrix
    
    def sample_copula(self, n_samples, correlation_matrix=None):
        """Sample from fitted Gaussian copula"""
        if correlation_matrix is None:
            correlation_matrix = self.correlation_matrix
            
        # Sample from multivariate normal
        Z = np.random.multivariate_normal(
            mean=np.zeros(correlation_matrix.shape[0]),
            cov=correlation_matrix,
            size=n_samples
        )
        
        # Transform to uniform
        U = norm.cdf(Z)
        
        return U
    
    def copula_density(self, u, correlation_matrix=None):
        """Compute Gaussian copula density"""
        if correlation_matrix is None:
            correlation_matrix = self.correlation_matrix
            
        # Transform to normal
        z = norm.ppf(np.clip(u, 1e-6, 1-1e-6))
        
        # Compute density
        inv_corr = np.linalg.inv(correlation_matrix)
        det_corr = np.linalg.det(correlation_matrix)
        
        density = (1/np.sqrt(det_corr)) * np.exp(
            -0.5 * np.sum(z * (inv_corr - np.eye(len(z))) @ z.T, axis=0)
        )
        
        return density

class CopulaLinearModel:
    """Linear model enhanced with Gaussian copula"""
    
    def __init__(self):
        self.copula = GaussianCopula()
        self.linear_models = {}
        self.copula_correlation = None
        
    def fit(self, X, y):
        """Fit copula-enhanced linear model"""
        # Combine X and y for copula fitting
        data = np.column_stack([X.flatten() if X.ndim > 1 else X, y])
        
        # Fit marginal distributions
        self.copula.fit_marginals(data)
        
        # Fit copula
        self.copula_correlation = self.copula.fit_copula(data)
        
        # Fit standard linear models for comparison
        X_reshaped = X.reshape(-1, 1) if X.ndim == 1 else X
        
        models = {
            'OLS': LinearRegression(),
            'Ridge': Ridge(alpha=0.5),
            'Lasso': Lasso(alpha=0.1, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
        }
        
        self.linear_models = {}
        for name, model in models.items():
            model.fit(X_reshaped, y)
            self.linear_models[name] = model
    
    def copula_enhanced_prediction(self, X, method='OLS'):
        """Make predictions using copula information"""
        X_reshaped = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Standard linear prediction
        base_pred = self.linear_models[method].predict(X_reshaped)
        
        # Transform input to uniform using copula
        X_uniform = self.copula.transform_to_uniform(
            np.column_stack([X.flatten(), base_pred])
        )
        
        # Get copula density weights
        try:
            copula_weights = self.copula.copula_density(X_uniform)
            copula_weights = copula_weights / np.mean(copula_weights)  # Normalize
        except:
            copula_weights = np.ones(len(base_pred))
        
        # Weight predictions by copula density
        enhanced_pred = base_pred * copula_weights
        
        return enhanced_pred, base_pred, copula_weights
    
    def simulate_from_copula(self, n_samples):
        """Simulate new data from fitted copula"""
        if self.copula_correlation is None:
            raise ValueError("Model must be fitted first")
            
        # Sample from copula
        U = self.copula.sample_copula(n_samples, self.copula_correlation)
        
        # Transform back to original marginals (approximate)
        simulated_data = np.zeros_like(U)
        
        for i in range(U.shape[1]):
            # Use empirical quantile function (inverse CDF)
            quantiles = U[:, i]
            data = self.copula.marginal_cdfs[i]['data']
            indices = (quantiles * len(data)).astype(int)
            indices = np.clip(indices, 0, len(data) - 1)
            simulated_data[:, i] = data[indices]
        
        return simulated_data[:, 0], simulated_data[:, 1]

class CopulaModelTester:
    """Test copula-enhanced linear models"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.copula_model = CopulaLinearModel()
        self.results = {}
        
    def run_copula_analysis(self):
        """Run comprehensive copula analysis"""
        print("Fitting Gaussian copula...")
        self.copula_model.fit(self.X, self.y)
        
        results = {}
        methods = ['OLS', 'Ridge', 'Lasso', 'ElasticNet']
        
        for method in methods:
            print(f"Testing {method} with copula enhancement...")
            
            # Standard prediction
            X_reshaped = self.X.reshape(-1, 1) if self.X.ndim == 1 else self.X
            standard_pred = self.copula_model.linear_models[method].predict(X_reshaped)
            
            # Copula-enhanced prediction
            try:
                enhanced_pred, base_pred, weights = self.copula_model.copula_enhanced_prediction(
                    self.X, method
                )
                
                # Calculate metrics for both
                results[f'{method}_Standard'] = {
                    'predictions': standard_pred,
                    'r2': r2_score(self.y, standard_pred),
                    'mse': mean_squared_error(self.y, standard_pred),
                    'mae': mean_absolute_error(self.y, standard_pred),
                    'type': 'standard'
                }
                
                results[f'{method}_Copula'] = {
                    'predictions': enhanced_pred,
                    'r2': r2_score(self.y, enhanced_pred),
                    'mse': mean_squared_error(self.y, enhanced_pred),
                    'mae': mean_absolute_error(self.y, enhanced_pred),
                    'weights': weights,
                    'type': 'copula'
                }
                
            except Exception as e:
                print(f"Copula enhancement failed for {method}: {e}")
                results[f'{method}_Standard'] = {
                    'predictions': standard_pred,
                    'r2': r2_score(self.y, standard_pred),
                    'mse': mean_squared_error(self.y, standard_pred),
                    'mae': mean_absolute_error(self.y, standard_pred),
                    'type': 'standard'
                }
        
        self.results = results
        return results
    
    def get_copula_statistics(self):
        """Get copula-specific statistics"""
        if self.copula_model.copula_correlation is not None:
            return {
                'copula_correlation': self.copula_model.copula_correlation,
                'linear_correlation': np.corrcoef(self.X, self.y)[0, 1],
                'kendall_tau': stats.kendalltau(self.X, self.y)[0],
                'spearman_rho': stats.spearmanr(self.X, self.y)[0]
            }
        return {}

def analyze_mean_reversion(asset):
    """Analyze mean reversion properties of an asset"""
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(asset)
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'mean': np.mean(asset),
            'std': np.std(asset),
            'autocorr_lag1': np.corrcoef(asset[:-1], asset[1:])[0,1]
        }
    except ImportError:
        return {
            'adf_statistic': None,
            'adf_pvalue': None,
            'is_stationary': None,
            'mean': np.mean(asset),
            'std': np.std(asset),
            'autocorr_lag1': np.corrcoef(asset[:-1], asset[1:])[0,1]
        }

def plot_copula_results(asset1, asset2, copula_results, copula_stats):
    """Create comprehensive plots including copula analysis"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # Plot 1: Time series of both assets
    axes[0,0].plot(asset1, label='Asset 1', alpha=0.8)
    axes[0,0].plot(asset2, label='Asset 2', alpha=0.8)
    axes[0,0].axhline(y=np.mean(asset1), color='blue', linestyle='--', alpha=0.5)
    axes[0,0].axhline(y=np.mean(asset2), color='orange', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Mean-Reverting Assets Over Time')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of assets
    axes[0,1].scatter(asset1, asset2, alpha=0.6, s=15)
    axes[0,1].set_title('Asset Relationship')
    axes[0,1].set_xlabel('Asset 1')
    axes[0,1].set_ylabel('Asset 2')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Copula correlation matrix heatmap
    if 'copula_correlation' in copula_stats:
        corr_matrix = copula_stats['copula_correlation']
        im = axes[0,2].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        axes[0,2].set_title('Gaussian Copula Correlation')
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                axes[0,2].text(j, i, f'{corr_matrix[i,j]:.3f}', 
                              ha='center', va='center')
        plt.colorbar(im, ax=axes[0,2])
    
    # Plot 4: Performance comparison - R²
    standard_methods = [k for k in copula_results.keys() if 'Standard' in k]
    copula_methods = [k for k in copula_results.keys() if 'Copula' in k]
    
    method_names = [m.replace('_Standard', '') for m in standard_methods]
    standard_r2 = [copula_results[m]['r2'] for m in standard_methods]
    copula_r2 = [copula_results[m]['r2'] for m in copula_methods if m in copula_results]
    
    x_pos = np.arange(len(method_names))
    width = 0.35
    
    axes[1,0].bar(x_pos - width/2, standard_r2, width, label='Standard', alpha=0.7)
    if copula_r2:
        axes[1,0].bar(x_pos + width/2, copula_r2, width, label='Copula Enhanced', alpha=0.7)
    axes[1,0].set_title('R² Score Comparison')
    axes[1,0].set_xlabel('Methods')
    axes[1,0].set_ylabel('R² Score')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(method_names, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: MSE comparison
    standard_mse = [copula_results[m]['mse'] for m in standard_methods]
    copula_mse = [copula_results[m]['mse'] for m in copula_methods if m in copula_results]
    
    axes[1,1].bar(x_pos - width/2, standard_mse, width, label='Standard', alpha=0.7)
    if copula_mse:
        axes[1,1].bar(x_pos + width/2, copula_mse, width, label='Copula Enhanced', alpha=0.7)
    axes[1,1].set_title('MSE Comparison')
    axes[1,1].set_xlabel('Methods')
    axes[1,1].set_ylabel('MSE')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(method_names, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Correlation comparison
    if copula_stats:
        corr_types = ['Linear', 'Kendall Tau', 'Spearman Rho', 'Copula']
        corr_values = [
            copula_stats.get('linear_correlation', 0),
            copula_stats.get('kendall_tau', 0),
            copula_stats.get('spearman_rho', 0),
            copula_stats.get('copula_correlation', [[0, 0], [0, 0]])[0, 1]
        ]
        
        axes[1,2].bar(corr_types, corr_values, alpha=0.7)
        axes[1,2].set_title('Correlation Measures')
        axes[1,2].set_ylabel('Correlation')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].grid(True, alpha=0.3)
    
    # Plot 7: Uniform transformation (copula space)
    try:
        from scipy.stats import rankdata
        u1 = rankdata(asset1) / (len(asset1) + 1)
        u2 = rankdata(asset2) / (len(asset2) + 1)
        axes[2,0].scatter(u1, u2, alpha=0.6, s=15)
        axes[2,0].set_title('Data in Copula Space (Uniform Marginals)')
        axes[2,0].set_xlabel('U1 (Asset 1 ranks)')
        axes[2,0].set_ylabel('U2 (Asset 2 ranks)')
        axes[2,0].grid(True, alpha=0.3)
    except:
        axes[2,0].text(0.5, 0.5, 'Copula transformation\nnot available', 
                       ha='center', va='center', transform=axes[2,0].transAxes)
    
    # Plot 8: Prediction comparison for best method
    if copula_methods and standard_methods:
        best_standard = max(standard_methods, key=lambda x: copula_results[x]['r2'])
        best_copula = best_standard.replace('Standard', 'Copula')
        
        if best_copula in copula_results:
            axes[2,1].scatter(copula_results[best_standard]['predictions'], asset2, 
                             alpha=0.5, label='Standard', s=10)
            axes[2,1].scatter(copula_results[best_copula]['predictions'], asset2, 
                             alpha=0.5, label='Copula Enhanced', s=10)
            axes[2,1].plot([asset2.min(), asset2.max()], [asset2.min(), asset2.max()], 
                          'r--', alpha=0.5)
            axes[2,1].set_title(f'Predictions Comparison - {best_standard.replace("_Standard", "")}')
            axes[2,1].set_xlabel('Predicted')
            axes[2,1].set_ylabel('Actual')
            axes[2,1].legend()
            axes[2,1].grid(True, alpha=0.3)
    
    # Plot 9: Copula weights (if available)
    if copula_methods:
        method_with_weights = None
        for method in copula_methods:
            if method in copula_results and 'weights' in copula_results[method]:
                method_with_weights = method
                break
        
        if method_with_weights:
            weights = copula_results[method_with_weights]['weights']
            axes[2,2].plot(weights, alpha=0.7)
            axes[2,2].set_title('Copula Density Weights')
            axes[2,2].set_xlabel('Observation')
            axes[2,2].set_ylabel('Weight')
            axes[2,2].grid(True, alpha=0.3)
        else:
            axes[2,2].text(0.5, 0.5, 'Copula weights\nnot available', 
                          ha='center', va='center', transform=axes[2,2].transAxes)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function with copula analysis"""
    print("Generating Mean-Reverting Synthetic Assets...")
    
    # Generate synthetic assets
    generator = MeanRevertingAssets(n_periods=1000)
    asset1, asset2 = generator.generate_assets()
    
    # Analyze mean reversion properties
    print("\n=== Mean Reversion Analysis ===")
    analysis1 = analyze_mean_reversion(asset1)
    analysis2 = analyze_mean_reversion(asset2)
    
    print(f"Asset 1: Mean={analysis1['mean']:.2f}, Std={analysis1['std']:.2f}")
    if analysis1['adf_pvalue'] is not None:
        print(f"         ADF p-value={analysis1['adf_pvalue']:.4f}, Stationary={analysis1['is_stationary']}")
    print(f"         Autocorrelation (lag1)={analysis1['autocorr_lag1']:.4f}")
    
    print(f"Asset 2: Mean={analysis2['mean']:.2f}, Std={analysis2['std']:.2f}")
    if analysis2['adf_pvalue'] is not None:
        print(f"         ADF p-value={analysis2['adf_pvalue']:.4f}, Stationary={analysis2['is_stationary']}")
    print(f"         Autocorrelation (lag1)={analysis2['autocorr_lag1']:.4f}")
    
    # Run copula analysis
    print("\n=== Copula-Enhanced Linear Model Analysis ===")
    tester = CopulaModelTester(asset1, asset2)
    copula_results = tester.run_copula_analysis()
    copula_stats = tester.get_copula_statistics()
    
    # Display results
    print("\n=== Model Comparison Results ===")
    print(f"{'Method':<20} {'R²':<8} {'MSE':<10} {'MAE':<10} {'Type':<10}")
    print("-" * 70)
    
    for method, result in copula_results.items():
        print(f"{method:<20} {result['r2']:<8.4f} {result['mse']:<10.2f} "
              f"{result['mae']:<10.2f} {result['type']:<10}")
    
    # Display copula statistics
    if copula_stats:
        print(f"\n=== Copula Statistics ===")
        print(f"Linear Correlation: {copula_stats.get('linear_correlation', 'N/A'):.4f}")
        print(f"Kendall's Tau: {copula_stats.get('kendall_tau', 'N/A'):.4f}")
        print(f"Spearman's Rho: {copula_stats.get('spearman_rho', 'N/A'):.4f}")
        if 'copula_correlation' in copula_stats:
            print(f"Copula Correlation: {copula_stats['copula_correlation'][0,1]:.4f}")
    
    # Create plots
    plot_copula_results(asset1, asset2, copula_results, copula_stats)
    
    # Generate simulations from copula
    print("\n=== Generating Simulations from Fitted Copula ===")
    try:
        sim_asset1, sim_asset2 = tester.copula_model.simulate_from_copula(200)
        print(f"Generated {len(sim_asset1)} simulated data points")
        print(f"Simulated Asset 1: Mean={np.mean(sim_asset1):.2f}, Std={np.std(sim_asset1):.2f}")
        print(f"Simulated Asset 2: Mean={np.mean(sim_asset2):.2f}, Std={np.std(sim_asset2):.2f}")
        print(f"Simulated Correlation: {np.corrcoef(sim_asset1, sim_asset2)[0,1]:.4f}")
    except Exception as e:
        print(f"Simulation failed: {e}")
    
    return asset1, asset2, copula_results, copula_stats

if __name__ == "__main__":
    # Run the analysis
    asset1, asset2, results, stats = main()