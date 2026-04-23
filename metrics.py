import numpy as np


def calculate_metrics(returns_array, initial_capital=100.0, risk_free_rate=0.02):
    """
    Calculate CAGR, max drawdown, volatility, final capital, and Sharpe ratio.

    returns_array: 1D numpy array of daily returns.
    """
    if len(returns_array) == 0:
        return {
            "cagr": 0,
            "max_dd": 0,
            "vol": 0,
            "sharpe": 0,
            "final_cap": initial_capital,
            "min_cap": initial_capital,
            "max_cap": initial_capital,
            "capital_curve": np.array([initial_capital]),
        }

    capital = initial_capital * np.cumprod(1 + returns_array)
    final_cap = capital[-1]
    max_cap = np.max(capital)
    min_cap = np.min(capital)

    total_days = len(returns_array)
    cagr = (final_cap / initial_capital) ** (252.0 / total_days) - 1.0

    peak = np.maximum.accumulate(capital)
    drawdown = (peak - capital) / peak
    max_dd = np.max(drawdown)

    vol = np.std(returns_array) * np.sqrt(252.0)
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0

    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "vol": vol,
        "sharpe": sharpe,
        "final_cap": final_cap,
        "min_cap": min_cap,
        "max_cap": max_cap,
        "capital_curve": capital,
    }
