"""
DQI Multi-Factor Equity Framework — Configuration
Adapted from DQI_version 2 to work with PROMETHEUS .parquet data.
Single source of truth for all parameters.
"""
import os

# ── Factor weights (must sum to 1.0) ────────────────────────────────────────
FACTOR_WEIGHTS = {
    'value':         0.20,
    'quality':       0.20,
    'growth':        0.15,
    'momentum':      0.20,
    'revisions':     0.10,
    'prof_momentum': 0.15,
}

# ── Sub-factor weights per factor (must sum to 1.0 within each) ─────────────
SUBFACTOR_WEIGHTS = {
    'value': {
        'ev_ebitda':      0.40,   # inverted: lower = better
        'fcf_yield':      0.35,
        'earnings_yield': 0.25,
    },
    'quality': {
        'roic':            0.40,
        'net_debt_ebitda': 0.30,  # inverted: lower = better
        'gross_margin':    0.30,
    },
    'growth': {
        'rev_growth_ttm': 0.40,
        'eps_growth_fy1': 0.40,
        'rev_growth_ntm': 0.20,   # analyst NTM estimate
    },
    'momentum': {
        'return_12m_ex1': 0.40,   # Jegadeesh-Titman
        'return_6m':      0.30,
        'return_12m':     0.30,
    },
    'revisions': {
        'eps_revision_1m': 1.00,
    },
    'prof_momentum': {
        'delta_roic':         0.50,
        'delta_gross_margin': 0.30,
        'delta_oper_margin':  0.20,
    },
}

# Sub-factors where LOWER value = BETTER (negated before z-scoring)
INVERT_SUBFACTORS = {'ev_ebitda', 'net_debt_ebitda'}

# ── Scoring pipeline parameters ──────────────────────────────────────────────
WINSORIZE_PCT          = (0.01, 0.99)  # 1st/99th percentile
MIN_WINSORIZE_SAMPLE   = 20            # min valid values to winsorize
MIN_SECTOR_SIZE        = 5             # min stocks per sector for sector-neutral z-score
MIN_COMPOSITE_COVERAGE = 0.70          # min factor weight coverage to compute composite
MAX_FACTOR_WEIGHT_MULT = 1.50          # max re-normalization multiplier (caps weight drift)

# ── Quintile rating bands (cumulative percentile thresholds) ─────────────────
RATING_BANDS = [
    (0.80, 'STRONG'),
    (0.60, 'GOOD'),
    (0.40, 'NEUTRAL'),
    (0.20, 'WEAK'),
    (0.00, 'POOR'),
]

# ── Regime-based factor weights (switched via VIX) ───────────────────────────
VIX_REGIME_THRESHOLDS = {'CALM': 15, 'NORMAL': 25}  # STRESS = above 25

REGIME_FACTOR_WEIGHTS = {
    'CALM': {
        'value': 0.15, 'quality': 0.15, 'growth': 0.20,
        'momentum': 0.25, 'revisions': 0.10, 'prof_momentum': 0.15,
    },
    'NORMAL': {  # matches FACTOR_WEIGHTS above
        'value': 0.20, 'quality': 0.20, 'growth': 0.15,
        'momentum': 0.20, 'revisions': 0.10, 'prof_momentum': 0.15,
    },
    'STRESS': {
        'value': 0.25, 'quality': 0.30, 'growth': 0.10,
        'momentum': 0.10, 'revisions': 0.10, 'prof_momentum': 0.15,
    },
}

# ── Financial derivation parameters ─────────────────────────────────────────
ASSUMED_TAX_RATE    = 0.21    # US corporate tax rate proxy (for ROIC)
EBIT_TO_EBITDA_MULT = 1.15   # EBITDA ≈ EBIT × 1.15 (proxy when D&A not available)

# ── Momentum lookback windows ─────────────────────────────────────────────────
MOMENTUM_LOOKBACK = {
    'return_12m_ex1_skip':   1,   # skip last N months (Jegadeesh-Titman ex-1m)
    'return_6m_months':      6,
    'return_12m_months':    12,
}

# ── EPS revision window ───────────────────────────────────────────────────────
EPS_REVISION_WEEKS = 4   # compare consensus EPS N weeks ago vs today

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'PROMETHEUS', 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

PARQUET_FILES = {
    'metadata':          os.path.join(DATA_DIR, 'metadata.parquet'),
    'prices':            os.path.join(DATA_DIR, 'prices.parquet'),
    'sp500_prices':      os.path.join(DATA_DIR, 'sp500_prices.parquet'),
    'vix':               os.path.join(DATA_DIR, 'vix_time_series.parquet'),
    'financials':        os.path.join(DATA_DIR, 'financials.parquet'),
    'analyst_estimates': os.path.join(DATA_DIR, 'analyst_estimates.parquet'),
}

OUTPUT_FILES = {
    'scores_parquet': os.path.join(OUTPUT_DIR, 'dqi_scores.parquet'),
    'scores_csv':     os.path.join(OUTPUT_DIR, 'dqi_scores.csv'),
}
