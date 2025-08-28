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
        """Generate Ornstein-Uhlenbeck process"""
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
            ranks = rankdata(X[:, i], method='average')
            U[:, i] = ranks / (len(ranks) + 1)
            
        return U
    
    def transform_to_normal(self, U):
        """Transform uniform marginals to standard normal"""
        U_clipped = np.clip(U, 1e-6, 1-1e-6)
        return norm.ppf(U_clipped)
    
    def fit_copula(self, X):
        """Fit Gaussian copula to the data"""
        U = self.transform_to_uniform(X)
        Z = self.transform_to_normal(U)
        self.correlation_matrix = np.corrcoef(Z.T)
        return self.correlation_matrix
    
    def copula_density(self, u):
        """Compute Gaussian copula density"""
        z = norm.ppf(np.clip(u, 1e-6, 1-1e-6))
        inv_corr = np.linalg.inv(self.correlation_matrix)
        det_corr = np.linalg.det(self.correlation_matrix)
        
        density = (1/np.sqrt(det_corr)) * np.exp(
            -0.5 * np.sum(z * (inv_corr - np.eye(len(z))) @ z.T, axis=0)
        )
        return density

class ZScoreTradingStrategy:
    """Classical z-score based pairs trading strategy"""
    
    def __init__(self, window=30, z_entry=2.0, z_exit=0.5, z_stop=3.5):
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.spread = None
        self.z_scores = None
        self.hedge_ratio = None
        
    def calculate_spread(self, asset1, asset2):
        """Calculate spread using rolling linear regression"""
        n = len(asset1)
        self.spread = np.full(n, np.nan)
        self.hedge_ratio = np.full(n, np.nan)
        
        for i in range(self.window, n):
            # Use rolling window for regression
            X = asset1[i-self.window:i].reshape(-1, 1)
            y = asset2[i-self.window:i]
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Store hedge ratio
            self.hedge_ratio[i] = model.coef_[0]
            
            # Calculate spread: asset2 - hedge_ratio * asset1
            self.spread[i] = asset2[i] - model.coef_[0] * asset1[i] - model.intercept_
        
        return self.spread
    
    def calculate_z_scores(self):
        """Calculate rolling z-scores of the spread"""
        if self.spread is None:
            raise ValueError("Must calculate spread first")
            
        self.z_scores = np.full(len(self.spread), np.nan)
        
        for i in range(self.window * 2, len(self.spread)):
            window_spread = self.spread[i-self.window:i]
            window_spread = window_spread[~np.isnan(window_spread)]
            
            if len(window_spread) > 10:
                mean_spread = np.mean(window_spread)
                std_spread = np.std(window_spread)
                if std_spread > 0:
                    self.z_scores[i] = (self.spread[i] - mean_spread) / std_spread
        
        return self.z_scores
    
    def generate_signals(self, asset1, asset2):
        """Generate trading signals based on z-scores"""
        spread = self.calculate_spread(asset1, asset2)
        z_scores = self.calculate_z_scores()
        
        signals = np.zeros(len(asset1))
        positions = np.zeros(len(asset1))
        current_position = 0
        entry_count = 0
        exit_count = 0
        
        for i in range(1, len(z_scores)):
            positions[i] = current_position  # Carry forward position
            
            if np.isnan(z_scores[i]):
                continue
                
            # Entry signals
            if current_position == 0:
                if z_scores[i] > self.z_entry:  # Spread is high, expect it to decrease
                    current_position = -1  # Short spread
                    signals[i] = -1
                    entry_count += 1
                    if entry_count <= 3:
                        print(f"Z-score short entry at {i}: z={z_scores[i]:.3f} > {self.z_entry}")
                elif z_scores[i] < -self.z_entry:  # Spread is low, expect it to increase
                    current_position = 1   # Long spread
                    signals[i] = 1
                    entry_count += 1
                    if entry_count <= 3:
                        print(f"Z-score long entry at {i}: z={z_scores[i]:.3f} < {-self.z_entry}")
            
            # Exit signals
            elif current_position != 0:
                if (abs(z_scores[i]) < self.z_exit or  # Normal exit (mean reversion)
                    abs(z_scores[i]) > self.z_stop):   # Stop loss
                    signals[i] = -current_position  # Close position
                    current_position = 0
                    exit_count += 1
                    if exit_count <= 3:
                        print(f"Z-score exit at {i}: z={z_scores[i]:.3f}")
            
            positions[i] = current_position
        
        print(f"Z-score strategy: {entry_count} entries, {exit_count} exits")
        return signals, positions, z_scores

class CopulaTradingStrategy:
    """Copula-based pairs trading strategy"""
    
    def __init__(self, window=50, prob_entry=0.95, prob_exit=0.6, refit_freq=20):
        self.window = window
        self.prob_entry = prob_entry
        self.prob_exit = prob_exit
        self.refit_freq = refit_freq
        self.copula = GaussianCopula()
        
    def calculate_conditional_probabilities(self, asset1, asset2):
        """Calculate conditional probabilities using copula"""
        n = len(asset1)
        cond_probs = np.full(n, np.nan)
        
        # Use overlapping windows for more frequent updates
        for i in range(self.window, n):
            # Fit copula on rolling window
            window_data = np.column_stack([
                asset1[i-self.window:i], 
                asset2[i-self.window:i]
            ])
            
            try:
                # Fit copula on historical window
                self.copula.fit_marginals(window_data)
                corr_matrix = self.copula.fit_copula(window_data)
                
                if corr_matrix is not None and not np.isnan(corr_matrix).any():
                    # Get current observation ranks within the historical window
                    current_vals = np.array([asset1[i], asset2[i]])
                    
                    # Calculate empirical percentiles within the window
                    u1 = (np.sum(window_data[:, 0] <= current_vals[0]) + 0.5) / (len(window_data) + 1)
                    u2 = (np.sum(window_data[:, 1] <= current_vals[1]) + 0.5) / (len(window_data) + 1)
                    
                    # Ensure u values are in valid range
                    u1 = np.clip(u1, 0.01, 0.99)
                    u2 = np.clip(u2, 0.01, 0.99)
                    
                    # Transform to normal space
                    z1 = norm.ppf(u1)
                    z2 = norm.ppf(u2)
                    
                    # Get copula correlation
                    rho = corr_matrix[0, 1]
                    
                    # Calculate conditional probability P(U2 <= u2 | U1 = u1)
                    # For Gaussian copula: Z2|Z1 ~ N(rho*Z1, 1-rho²)
                    if abs(rho) > 0.01:  # Avoid division by zero
                        conditional_mean = rho * z1
                        conditional_std = np.sqrt(max(1 - rho**2, 0.01))
                        
                        # P(Z2 <= z2 | Z1 = z1)
                        cond_prob = norm.cdf((z2 - conditional_mean) / conditional_std)
                        cond_probs[i] = cond_prob
                    else:
                        # If no correlation, use marginal probability
                        cond_probs[i] = u2
                        
            except Exception as e:
                if i == self.window:  # Only print error for first occurrence
                    print(f"Copula fitting failed: {e}")
                # Use simple percentile as fallback
                try:
                    window_asset2 = asset2[i-self.window:i]
                    cond_probs[i] = (np.sum(window_asset2 <= asset2[i]) + 0.5) / (len(window_asset2) + 1)
                except:
                    continue
        
        return cond_probs
    
    def generate_signals(self, asset1, asset2):
        """Generate trading signals based on copula conditional probabilities"""
        cond_probs = self.calculate_conditional_probabilities(asset1, asset2)
        
        signals = np.zeros(len(asset1))
        positions = np.zeros(len(asset1))
        current_position = 0
        
        # Add debug information
        valid_probs = cond_probs[~np.isnan(cond_probs)]
        print(f"Copula strategy: {len(valid_probs)} valid probabilities out of {len(cond_probs)}")
        if len(valid_probs) > 0:
            print(f"Probability range: [{np.min(valid_probs):.3f}, {np.max(valid_probs):.3f}]")
            print(f"Mean probability: {np.mean(valid_probs):.3f}")
        
        entry_count = 0
        exit_count = 0
        
        for i in range(1, len(cond_probs)):
            positions[i] = current_position  # Carry forward position
            
            if np.isnan(cond_probs[i]):
                continue
            
            prob = cond_probs[i]
            
            # Entry signals based on extreme conditional probabilities
            if current_position == 0:
                if prob > self.prob_entry:  # Asset2 overvalued given Asset1
                    current_position = -1  # Short asset2, long asset1
                    signals[i] = -1
                    entry_count += 1
                    if entry_count <= 5:  # Debug first few entries
                        print(f"Short entry at {i}: prob={prob:.3f} > {self.prob_entry}")
                elif prob < (1 - self.prob_entry):  # Asset2 undervalued given Asset1
                    current_position = 1   # Long asset2, short asset1
                    signals[i] = 1
                    entry_count += 1
                    if entry_count <= 5:  # Debug first few entries
                        print(f"Long entry at {i}: prob={prob:.3f} < {1-self.prob_entry}")
            
            # Exit signals based on probability convergence
            elif current_position != 0:
                # Exit when probability is between exit thresholds (closer to 0.5)
                if (1 - self.prob_exit) <= prob <= self.prob_exit:
                    signals[i] = -current_position  # Close position
                    current_position = 0
                    exit_count += 1
                    if exit_count <= 5:  # Debug first few exits
                        print(f"Exit at {i}: prob={prob:.3f} in [{1-self.prob_exit}, {self.prob_exit}]")
            
            positions[i] = current_position
        
        print(f"Copula strategy generated {entry_count} entries and {exit_count} exits")
        print(f"Final position status: {np.sum(positions != 0)} periods in position")
        
        return signals, positions, cond_probs

class TradingBacktest:
    """Comprehensive backtesting framework for trading strategies"""
    
    def __init__(self, transaction_cost=0.001):
        self.transaction_cost = transaction_cost
        
    def calculate_returns(self, signals, asset1_prices, asset2_prices, positions):
        """Calculate strategy returns including transaction costs"""
        n = len(signals)
        strategy_returns = np.zeros(n)
        
        # Calculate simple returns (not log returns for better interpretation)
        asset1_returns = np.diff(asset1_prices) / asset1_prices[:-1]
        asset2_returns = np.diff(asset2_prices) / asset2_prices[:-1]
        
        # Pad with zero for first observation
        asset1_returns = np.concatenate([[0], asset1_returns])
        asset2_returns = np.concatenate([[0], asset2_returns])
        
        for i in range(1, n):
            if positions[i] != 0:
                # Position interpretation:
                # +1 means long spread = long asset2, short asset1
                # -1 means short spread = short asset2, long asset1
                
                # Calculate P&L for the spread trade
                if positions[i] == 1:  # Long spread
                    # Profit when asset2 outperforms asset1
                    daily_return = asset2_returns[i] - asset1_returns[i]
                else:  # Short spread (positions[i] == -1)
                    # Profit when asset1 outperforms asset2
                    daily_return = asset1_returns[i] - asset2_returns[i]
                
                # Apply transaction costs when position changes
                if signals[i] != 0:  # Trade occurred
                    daily_return -= self.transaction_cost
                
                strategy_returns[i] = daily_return
        
        return strategy_returns
    
    def calculate_metrics(self, returns):
        """Calculate comprehensive trading metrics"""
        # Only consider non-zero returns for active trading periods
        active_returns = returns[returns != 0]
        
        if len(active_returns) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0
            }
        
        # Calculate cumulative performance
        cumulative_returns = np.cumsum(returns)
        total_return = cumulative_returns[-1]
        
        # Annualized metrics (assuming daily data)
        trading_days = len(active_returns)
        if trading_days > 0:
            annualized_return = np.mean(active_returns) * 252
            volatility = np.std(active_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        else:
            annualized_return = volatility = sharpe_ratio = 0
        
        # Calculate maximum drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate and profit factor
        positive_returns = active_returns[active_returns > 0]
        negative_returns = active_returns[active_returns < 0]
        
        win_rate = len(positive_returns) / len(active_returns) if len(active_returns) > 0 else 0
        
        if len(negative_returns) > 0 and np.sum(negative_returns) < 0:
            profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns))
        else:
            profit_factor = np.inf if len(positive_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(active_returns)
        }
    
    def run_backtest(self, strategy, asset1, asset2, strategy_name):
        """Run complete backtest for a strategy"""
        print(f"Running backtest for {strategy_name}...")
        
        signals, positions, indicators = strategy.generate_signals(asset1, asset2)
        returns = self.calculate_returns(signals, asset1, asset2, positions)
        metrics = self.calculate_metrics(returns)
        
        return {
            'signals': signals,
            'positions': positions,
            'returns': returns,
            'metrics': metrics,
            'indicators': indicators,
            'name': strategy_name
        }

def plot_strategy_comparison(asset1, asset2, zscore_result, copula_result):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    
    # Plot 1: Asset prices
    axes[0,0].plot(asset1, label='Asset 1', alpha=0.8)
    axes[0,0].plot(asset2, label='Asset 2', alpha=0.8)
    axes[0,0].set_title('Asset Prices Over Time')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Z-scores and trading signals
    z_scores = zscore_result['indicators']
    axes[0,1].plot(z_scores, alpha=0.7, label='Z-Score')
    axes[0,1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Entry Threshold')
    axes[0,1].axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)
    axes[0,1].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Exit Threshold')
    axes[0,1].axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
    
    # Mark trading signals
    signals = zscore_result['signals']
    signal_times = np.where(signals != 0)[0]
    signal_values = z_scores[signal_times]
    axes[0,1].scatter(signal_times, signal_values, c='red', s=30, alpha=0.8, marker='o')
    
    axes[0,1].set_title('Z-Score Strategy Signals')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Copula conditional probabilities and signals
    cond_probs = copula_result['indicators']
    axes[1,0].plot(cond_probs, alpha=0.7, label='Conditional Probability')
    axes[1,0].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Entry Threshold')
    axes[1,0].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    axes[1,0].axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Exit Threshold')
    axes[1,0].axhline(y=0.4, color='green', linestyle='--', alpha=0.5)
    
    # Mark trading signals
    copula_signals = copula_result['signals']
    copula_signal_times = np.where(copula_signals != 0)[0]
    copula_signal_values = cond_probs[copula_signal_times]
    axes[1,0].scatter(copula_signal_times, copula_signal_values, c='blue', s=30, alpha=0.8, marker='s')
    
    axes[1,0].set_title('Copula Strategy Signals')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Positions comparison
    axes[1,1].plot(zscore_result['positions'], label='Z-Score Positions', alpha=0.7)
    axes[1,1].plot(copula_result['positions'], label='Copula Positions', alpha=0.7)
    axes[1,1].set_title('Positions Over Time')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 5: Cumulative returns
    zscore_cum_returns = np.cumsum(zscore_result['returns'])
    copula_cum_returns = np.cumsum(copula_result['returns'])
    
    axes[2,0].plot(zscore_cum_returns, label='Z-Score Strategy', alpha=0.8)
    axes[2,0].plot(copula_cum_returns, label='Copula Strategy', alpha=0.8)
    axes[2,0].set_title('Cumulative Returns Comparison')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # Plot 6: Rolling Sharpe ratio (30-day window)
    window = 30
    zscore_rolling_sharpe = []
    copula_rolling_sharpe = []
    
    for i in range(window, len(zscore_result['returns'])):
        zscore_window = zscore_result['returns'][i-window:i]
        copula_window = copula_result['returns'][i-window:i]
        
        zscore_window = zscore_window[zscore_window != 0]
        copula_window = copula_window[copula_window != 0]
        
        zscore_sharpe = (np.mean(zscore_window) * np.sqrt(252) / 
                        (np.std(zscore_window) * np.sqrt(252)) if len(zscore_window) > 0 and np.std(zscore_window) > 0 else 0)
        copula_sharpe = (np.mean(copula_window) * np.sqrt(252) / 
                        (np.std(copula_window) * np.sqrt(252)) if len(copula_window) > 0 and np.std(copula_window) > 0 else 0)
        
        zscore_rolling_sharpe.append(zscore_sharpe)
        copula_rolling_sharpe.append(copula_sharpe)
    
    x_axis = range(window, len(zscore_result['returns']))
    axes[2,1].plot(x_axis, zscore_rolling_sharpe, label='Z-Score Rolling Sharpe', alpha=0.8)
    axes[2,1].plot(x_axis, copula_rolling_sharpe, label='Copula Rolling Sharpe', alpha=0.8)
    axes[2,1].set_title('Rolling Sharpe Ratio (30-day)')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    # Plot 7: Return distribution
    zscore_nonzero = zscore_result['returns'][zscore_result['returns'] != 0]
    copula_nonzero = copula_result['returns'][copula_result['returns'] != 0]
    
    axes[3,0].hist(zscore_nonzero, bins=30, alpha=0.6, label='Z-Score Returns', density=True)
    axes[3,0].hist(copula_nonzero, bins=30, alpha=0.6, label='Copula Returns', density=True)
    axes[3,0].set_title('Return Distribution')
    axes[3,0].legend()
    axes[3,0].grid(True, alpha=0.3)
    
    # Plot 8: Performance metrics comparison
    metrics_names = ['Total Return', 'Ann. Return', 'Volatility', 'Sharpe Ratio', 'Max DD', 'Win Rate']
    zscore_vals = [
        zscore_result['metrics']['total_return'],
        zscore_result['metrics']['annualized_return'],
        zscore_result['metrics']['volatility'],
        zscore_result['metrics']['sharpe_ratio'],
        abs(zscore_result['metrics']['max_drawdown']),
        zscore_result['metrics']['win_rate']
    ]
    copula_vals = [
        copula_result['metrics']['total_return'],
        copula_result['metrics']['annualized_return'],
        copula_result['metrics']['volatility'],
        copula_result['metrics']['sharpe_ratio'],
        abs(copula_result['metrics']['max_drawdown']),
        copula_result['metrics']['win_rate']
    ]
    
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    
    axes[3,1].bar(x_pos - width/2, zscore_vals, width, label='Z-Score', alpha=0.7)
    axes[3,1].bar(x_pos + width/2, copula_vals, width, label='Copula', alpha=0.7)
    axes[3,1].set_title('Performance Metrics Comparison')
    axes[3,1].set_xticks(x_pos)
    axes[3,1].set_xticklabels(metrics_names, rotation=45)
    axes[3,1].legend()
    axes[3,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function with strategy comparison"""
    print("=== Mean-Reverting Assets Trading Strategy Comparison ===\n")
    
    # Generate synthetic assets
    print("Generating synthetic mean-reverting assets...")
    generator = MeanRevertingAssets(n_periods=1000)
    asset1, asset2 = generator.generate_assets()
    
    print(f"Generated {len(asset1)} observations")
    print(f"Asset 1: Mean={np.mean(asset1):.2f}, Std={np.std(asset1):.2f}")
    print(f"Asset 2: Mean={np.mean(asset2):.2f}, Std={np.std(asset2):.2f}")
    print(f"Correlation: {np.corrcoef(asset1, asset2)[0,1]:.4f}\n")
    
    # Initialize strategies
    zscore_strategy = ZScoreTradingStrategy(
        window=30, z_entry=2.0, z_exit=0.5, z_stop=3.5
    )
    
    copula_strategy = CopulaTradingStrategy(
        window=40, prob_entry=0.85, prob_exit=0.65, refit_freq=1
    )
    
    # Initialize backtester
    backtester = TradingBacktest(transaction_cost=0.001)
    
    # Run backtests
    print("=== Running Strategy Backtests ===")
    zscore_result = backtester.run_backtest(zscore_strategy, asset1, asset2, "Z-Score")
    copula_result = backtester.run_backtest(copula_strategy, asset1, asset2, "Copula")
    
    # Display results
    print("\n=== Strategy Performance Comparison ===")
    print(f"{'Metric':<20} {'Z-Score':<15} {'Copula':<15} {'Difference':<15}")
    print("-" * 70)
    
    metrics_comparison = [
        ('Total Return', 'total_return'),
        ('Annualized Return', 'annualized_return'),
        ('Volatility', 'volatility'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown', 'max_drawdown'),
        ('Win Rate', 'win_rate'),
        ('Profit Factor', 'profit_factor'),
        ('Number of Trades', 'num_trades')
    ]
    
    for metric_name, metric_key in metrics_comparison:
        zscore_val = zscore_result['metrics'][metric_key]
        copula_val = copula_result['metrics'][metric_key]
        difference = copula_val - zscore_val
        
        if metric_key in ['win_rate']:
            print(f"{metric_name:<20} {zscore_val:<15.1%} {copula_val:<15.1%} {difference:<15.1%}")
        elif metric_key in ['profit_factor']:
            zscore_str = f"{zscore_val:.2f}" if zscore_val != np.inf else "∞"
            copula_str = f"{copula_val:.2f}" if copula_val != np.inf else "∞"
            diff_str = f"{difference:.2f}" if difference != np.inf and not np.isnan(difference) else "N/A"
            print(f"{metric_name:<20} {zscore_str:<15} {copula_str:<15} {diff_str:<15}")
        elif metric_key in ['num_trades']:
            print(f"{metric_name:<20} {zscore_val:<15.0f} {copula_val:<15.0f} {difference:<15.0f}")
        else:
            print(f"{metric_name:<20} {zscore_val:<15.4f} {copula_val:<15.4f} {difference:<15.4f}")
    
    # Strategy analysis
    print(f"\n=== Strategy Analysis ===")
    print(f"Z-Score Strategy:")
    print(f"  - Uses {zscore_strategy.window}-period rolling window for spread calculation")
    print(f"  - Entry threshold: ±{zscore_strategy.z_entry} standard deviations")
    print(f"  - Exit threshold: ±{zscore_strategy.z_exit} standard deviations")
    print(f"  - Stop loss: ±{zscore_strategy.z_stop} standard deviations")
    
    print(f"\nCopula Strategy:")
    print(f"  - Uses {copula_strategy.window}-period rolling window for copula fitting")
    print(f"  - Entry threshold: {copula_strategy.prob_entry:.1%} conditional probability")
    print(f"  - Exit threshold: {copula_strategy.prob_exit:.1%} conditional probability")
    print(f"  - Refits copula every {copula_strategy.refit_freq} periods")
    
    # Determine winner
    print(f"\n=== Summary ===")
    if zscore_result['metrics']['sharpe_ratio'] > copula_result['metrics']['sharpe_ratio']:
        print("🏆 Z-Score strategy outperformed based on Sharpe ratio")
        winner = "Z-Score"
    else:
        print("🏆 Copula strategy outperformed based on Sharpe ratio")
        winner = "Copula"
    
    print(f"Winner: {winner}")
    print(f"Sharpe Ratio Difference: {abs(zscore_result['metrics']['sharpe_ratio'] - copula_result['metrics']['sharpe_ratio']):.4f}")
    
    # Create comparison plots
    plot_strategy_comparison(asset1, asset2, zscore_result, copula_result)
    
    return {
        'asset1': asset1,
        'asset2': asset2,
        'zscore_result': zscore_result,
        'copula_result': copula_result
    }

if __name__ == "__main__":
    results = main()