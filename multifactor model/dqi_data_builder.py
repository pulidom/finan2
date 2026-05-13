"""
DQI Data Builder — PROMETHEUS Parquet Edition
=============================================
Carga todos los .parquet de PROMETHEUS/Data y construye un dict "cache"
con la misma estructura que espera dqi_factor_scores.py.

    cache[ticker] = {
        'gics_sub_industry_name': str,   # usado para sector-neutral z-score
        'sector': str,
        'name': str,
        # ── Value ──────────────────────────────────────────────────────────
        'ev_ebitda':       float,   # EV / TTM EBITDA  (lower=better → inverted)
        'fcf_yield':       float,   # TTM FCF / Mkt Cap
        'earnings_yield':  float,   # TTM Net Income / Mkt Cap  (inverse P/E)
        # ── Quality ────────────────────────────────────────────────────────
        'roic':            float,   # EBIT*(1−t) / Invested Capital
        'net_debt_ebitda': float,   # Net Debt / TTM EBITDA (lower=better → inverted)
        'gross_margin':    float,   # TTM Gross Profit / Revenue
        # ── Growth ─────────────────────────────────────────────────────────
        'rev_growth_ttm':  float,   # TTM Revenue YoY
        'eps_growth_fy1':  float,   # NTM EPS (analyst) / TTM EPS − 1
        'rev_growth_ntm':  float,   # NTM Revenue (analyst) / TTM Revenue − 1
        # ── Momentum ───────────────────────────────────────────────────────
        'return_12m_ex1':  float,   # price return [−12m → −1m]  Jegadeesh-Titman
        'return_6m':       float,   # price return last 6 months
        'return_12m':      float,   # price return last 12 months
        # ── Revisions ──────────────────────────────────────────────────────
        'eps_revision_1m': float,   # % change consensus EPS over last 4 weeks
        # ── Profitability Momentum ──────────────────────────────────────────
        'delta_roic':         float,
        'delta_gross_margin': float,
        'delta_oper_margin':  float,
        # ── Market meta ────────────────────────────────────────────────────
        'px_last': float,
        'mkt_cap': float,
    }

Uso:
    from dqi_data_builder import build_cache
    cache = build_cache()
"""

import warnings
import numpy as np
import pandas as pd

from dqi_config import (
    PARQUET_FILES,
    ASSUMED_TAX_RATE,
    MOMENTUM_LOOKBACK,
    EPS_REVISION_WEEKS,
    EBIT_TO_EBITDA_MULT,
)

warnings.filterwarnings('ignore', category=FutureWarning)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """División segura: ceros e infinitos → NaN."""
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _nearest_past_date(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    """Fecha disponible más cercana que sea ≤ target."""
    available = index[index <= target]
    return available[-1] if len(available) else index[0]


def _scalar(series: pd.Series, key: str):
    """Retorna float o None de forma segura."""
    try:
        v = series[key]
        return float(v) if pd.notna(v) else None
    except (KeyError, TypeError, ValueError):
        return None


# ─── TTM Financials ───────────────────────────────────────────────────────────

def _build_ttm(fin: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing 12-month aggregates por ticker (suma de últimas 4 quarters).
    También calcula:
      - rev_growth_ttm: TTM actual vs TTM un año atrás
      - gross_margin, ebit_margin, roic, net_debt, invested_capital, fcf
    Índice = symbol.
    """
    fin_s = fin.sort_values(['symbol', 'date'])

    ttm = (
        fin_s.groupby('symbol').tail(4)
             .groupby('symbol')
             .agg(
                 ttm_revenue      =('revenue',              'sum'),
                 ttm_gross_profit =('gross_profit',         'sum'),
                 ttm_ebit         =('ebit',                 'sum'),
                 ttm_net_income   =('net_income',           'sum'),
                 ttm_operating_cf =('operating_cf',         'sum'),
                 ttm_capex        =('capex_total',          'sum'),
                 total_assets     =('total_assets',         'last'),
                 long_term_debt   =('long_term_debt',       'last'),
                 short_term_debt  =('short_term_debt',      'last'),
                 cash             =('cash_and_equivalents', 'last'),
                 common_equity    =('common_equity',        'last'),
             )
    )

    # Derived TTM metrics
    ttm['fcf']         = ttm['ttm_operating_cf'] - ttm['ttm_capex'].fillna(0)
    ttm['gross_margin']= _safe_div(ttm['ttm_gross_profit'], ttm['ttm_revenue'])
    ttm['ebit_margin'] = _safe_div(ttm['ttm_ebit'],         ttm['ttm_revenue'])
    ttm['ttm_ebitda']  = ttm['ttm_ebit'] * EBIT_TO_EBITDA_MULT   # EBITDA proxy

    ttm['net_debt'] = (
        ttm['long_term_debt'].fillna(0)
        + ttm['short_term_debt'].fillna(0)
        - ttm['cash'].fillna(0)
    )
    ttm['invested_capital'] = (
        ttm['common_equity']
        + ttm['long_term_debt'].fillna(0)
        + ttm['short_term_debt'].fillna(0)
        - ttm['cash'].fillna(0)
    )
    ttm['roic'] = _safe_div(
        ttm['ttm_ebit'] * (1 - ASSUMED_TAX_RATE),
        ttm['invested_capital']
    )

    # Revenue YoY TTM: últimas 4Q vs las 4Q anteriores
    ttm_now  = fin_s.groupby('symbol').tail(4).groupby('symbol')['revenue'].sum()
    ttm_ago  = (
        fin_s.groupby('symbol', group_keys=False)
             .apply(
                 lambda x: x.iloc[-8:-4]['revenue'].sum() if len(x) >= 8 else np.nan,
                 include_groups=False,
             )
    )
    ttm['rev_growth_ttm'] = _safe_div(ttm_now, ttm_ago.replace(0, np.nan)) - 1

    return ttm


def _build_ttm_1y_ago(fin: pd.DataFrame) -> pd.DataFrame:
    """
    Métricas de hace un año (Q-5 a Q-8) para calcular deltas de profitabilidad.
    Índice = symbol.
    """
    fin_s = fin.sort_values(['symbol', 'date'])
    ttm_1y = (
        fin_s.groupby('symbol', group_keys=True)
             .apply(
                 lambda x: x.iloc[-8:-4] if len(x) >= 8 else pd.DataFrame(),
                 include_groups=True,
             )
    )
    if ttm_1y.empty:
        return pd.DataFrame()

    # After groupby-apply with group_keys=True the group label becomes level-0
    # of a MultiIndex; drop it so 'symbol' is a regular column again.
    if isinstance(ttm_1y.index, pd.MultiIndex):
        ttm_1y = ttm_1y.reset_index(level=0, drop=True)

    agg = (
        ttm_1y.groupby('symbol')
              .agg(
                  gp_1y  =('gross_profit', 'sum'),
                  ebit_1y=('ebit',         'sum'),
                  rev_1y =('revenue',      'sum'),
                  eq_1y  =('common_equity','last'),
                  ltd_1y =('long_term_debt','last'),
                  std_1y =('short_term_debt','last'),
                  cash_1y=('cash_and_equivalents','last'),
              )
    )
    agg['gross_margin_1y']    = _safe_div(agg['gp_1y'],   agg['rev_1y'])
    agg['ebit_margin_1y']     = _safe_div(agg['ebit_1y'], agg['rev_1y'])
    agg['invested_cap_1y']    = (
        agg['eq_1y']
        + agg['ltd_1y'].fillna(0)
        + agg['std_1y'].fillna(0)
        - agg['cash_1y'].fillna(0)
    )
    agg['roic_1y'] = _safe_div(
        agg['ebit_1y'] * (1 - ASSUMED_TAX_RATE),
        agg['invested_cap_1y']
    )
    return agg


# ─── Market Data & Momentum ───────────────────────────────────────────────────

def _build_market_data(prices: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Precio actual, market cap y returns de momentum por ticker.
    Índice = symbol.
    """
    prices_s = prices.sort_values('date')
    ref_date = prices_s['date'].max()
    dates    = prices_s['date'].unique()

    m1_date  = _nearest_past_date(dates, ref_date - pd.DateOffset(months=MOMENTUM_LOOKBACK['return_12m_ex1_skip']))
    m6_date  = _nearest_past_date(dates, ref_date - pd.DateOffset(months=MOMENTUM_LOOKBACK['return_6m_months']))
    m12_date = _nearest_past_date(dates, ref_date - pd.DateOffset(months=MOMENTUM_LOOKBACK['return_12m_months']))

    pivot = prices_s.pivot_table(index='date', columns='symbol', values='close_price')

    p_now = pivot.loc[ref_date]
    p_m1  = pivot.loc[m1_date]
    p_m6  = pivot.loc[m6_date]
    p_m12 = pivot.loc[m12_date]

    mkt = pd.DataFrame({
        'px_last':        p_now,
        'return_12m_ex1': p_m1  / p_m12 - 1,   # Jegadeesh-Titman: [−12m → −1m]
        'return_6m':      p_now / p_m6  - 1,
        'return_12m':     p_now / p_m12 - 1,
    })

    shares = meta.set_index('symbol')['shares_outstanding']
    mkt['mkt_cap'] = mkt['px_last'] * shares

    return mkt


# ─── Analyst Features ────────────────────────────────────────────────────────

def _build_analyst_features(est: pd.DataFrame,
                             ttm: pd.DataFrame,
                             meta: pd.DataFrame) -> pd.DataFrame:
    """
    Sub-factores derivados de las estimaciones de analistas.
    Índice = symbol.
    """
    est_s    = est.sort_values('date')
    ref_date = est_s['date'].max()
    cutoff   = ref_date - pd.DateOffset(weeks=EPS_REVISION_WEEKS)

    est_now = est_s[est_s['date'] == ref_date].set_index('symbol')
    est_4wk = est_s[est_s['date'] <= cutoff].groupby('symbol').last()

    # EPS revision: % cambio en consenso EPS últimas 4 semanas
    eps_revision_1m = _safe_div(
        est_now['eps'],
        est_4wk['eps'].reindex(est_now.index)
    ) - 1

    # NTM Revenue growth vs TTM
    ntm_rev = est_now['revenue'].reindex(ttm.index)
    rev_growth_ntm = _safe_div(ntm_rev, ttm['ttm_revenue'].replace(0, np.nan)) - 1

    # EPS growth FY1: EPS NTM (analista) vs EPS TTM
    shares  = meta.set_index('symbol')['shares_outstanding']
    ttm_eps = _safe_div(ttm['ttm_net_income'], shares.reindex(ttm.index))
    ntm_eps = est_now['eps'].reindex(ttm.index)
    eps_growth_fy1 = _safe_div(ntm_eps, ttm_eps.abs()) - 1

    return pd.DataFrame({
        'eps_revision_1m':        eps_revision_1m,
        'rev_growth_ntm':         rev_growth_ntm,
        'eps_growth_fy1':         eps_growth_fy1,
        'analyst_recommendation': est_now.get('recommendation', pd.Series(dtype=float)),
    })


# ─── Valuation ────────────────────────────────────────────────────────────────

def _build_valuation(ttm: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """EV/EBITDA, FCF yield, Earnings yield."""
    mkt_cap = mkt['mkt_cap']
    ev      = mkt_cap + ttm['net_debt']

    return pd.DataFrame({
        'ev_ebitda':      _safe_div(ev, ttm['ttm_ebitda']),
        'fcf_yield':      _safe_div(ttm['fcf'], mkt_cap),
        'earnings_yield': _safe_div(ttm['ttm_net_income'], mkt_cap),
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_cache(verbose: bool = True) -> dict:
    """
    Construye el cache completo desde los archivos .parquet de PROMETHEUS.

    Returns:
        cache: {ticker: {feature_name: value_or_None, ...}}
    """
    if verbose:
        print("Building DQI cache from PROMETHEUS parquet files...")

    # ── Load ─────────────────────────────────────────────────────────────────
    if verbose:
        print("  Loading parquet files...")
    dfs = {}
    for name, path in PARQUET_FILES.items():
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        dfs[name] = df
        if verbose:
            print(f"    {name:22s}: {len(df):>8,} rows")

    meta = dfs['metadata']
    fin  = dfs['financials']
    est  = dfs['analyst_estimates']
    px   = dfs['prices']
    vix  = dfs['vix']

    # ── Feature tables ───────────────────────────────────────────────────────
    if verbose: print("  Computing TTM financials...")
    ttm    = _build_ttm(fin)

    if verbose: print("  Computing 1Y-ago TTM for profitability deltas...")
    ttm_1y = _build_ttm_1y_ago(fin)

    if verbose: print("  Computing market data & momentum...")
    mkt    = _build_market_data(px, meta)

    if verbose: print("  Computing analyst features...")
    ana    = _build_analyst_features(est, ttm, meta)

    if verbose: print("  Computing valuation ratios...")
    val    = _build_valuation(ttm, mkt)

    # ── Profitability deltas ─────────────────────────────────────────────────
    delta_gm = ttm['gross_margin']  - ttm_1y['gross_margin_1y'].reindex(ttm.index)
    delta_em = ttm['ebit_margin']   - ttm_1y['ebit_margin_1y'].reindex(ttm.index)
    delta_rc = ttm['roic']          - ttm_1y['roic_1y'].reindex(ttm.index)

    # ── Latest VIX ───────────────────────────────────────────────────────────
    latest_vix = float(vix.sort_values('date')['close_price'].iloc[-1])

    # ── Universe: tickers con precios Y financials ───────────────────────────
    universe = sorted(set(ttm.index) & set(mkt.index))
    if verbose:
        print(f"  Universe: {len(universe)} tickers")

    meta_idx = meta.set_index('symbol')

    # ── Assemble cache ───────────────────────────────────────────────────────
    cache = {}
    for ticker in universe:
        def g(series, t=ticker):
            return _scalar(series, t)

        m = meta_idx.loc[ticker] if ticker in meta_idx.index else pd.Series(dtype=object)

        # net_debt_ebitda como serie para poder usar g()
        nde_series = _safe_div(ttm['net_debt'], ttm['ttm_ebitda'].replace(0, np.nan))

        cache[ticker] = {
            # Identifiers
            'name':                   str(m.get('company_name', '')),
            'sector':                 str(m.get('sector', '')),
            'gics_sub_industry_name': str(m.get('sub_industry', m.get('sector', ''))),
            'industry':               str(m.get('industry', '')),
            'currency':               str(m.get('currency', 'USD')),
            'exchange':               str(m.get('exchange', '')),
            # ── Value ────────────────────────────────────────────────────
            'ev_ebitda':       g(val['ev_ebitda']),
            'fcf_yield':       g(val['fcf_yield']),
            'earnings_yield':  g(val['earnings_yield']),
            # ── Quality ──────────────────────────────────────────────────
            'roic':            g(ttm['roic']),
            'net_debt_ebitda': g(nde_series),
            'gross_margin':    g(ttm['gross_margin']),
            # ── Growth ───────────────────────────────────────────────────
            'rev_growth_ttm':  g(ttm['rev_growth_ttm']),
            'eps_growth_fy1':  g(ana['eps_growth_fy1']),
            'rev_growth_ntm':  g(ana['rev_growth_ntm']),
            # ── Momentum ─────────────────────────────────────────────────
            'return_12m_ex1':  g(mkt['return_12m_ex1']),
            'return_6m':       g(mkt['return_6m']),
            'return_12m':      g(mkt['return_12m']),
            # ── Revisions ────────────────────────────────────────────────
            'eps_revision_1m': g(ana['eps_revision_1m']),
            # ── Profitability Momentum ────────────────────────────────────
            'delta_roic':         g(delta_rc),
            'delta_gross_margin': g(delta_gm),
            'delta_oper_margin':  g(delta_em),
            # ── Market ───────────────────────────────────────────────────
            'px_last':   g(mkt['px_last']),
            'mkt_cap':   g(mkt['mkt_cap']),
            'latest_vix': latest_vix,
            # ── Raw TTM (para referencia y output) ───────────────────────
            'ttm_revenue':    g(ttm['ttm_revenue']),
            'ttm_net_income': g(ttm['ttm_net_income']),
            'ttm_fcf':        g(ttm['fcf']),
            'ttm_ebitda':     g(ttm['ttm_ebitda']),
            'analyst_recommendation': g(ana['analyst_recommendation']),
        }

    if verbose:
        print(f"  Cache ready: {len(cache)} tickers.")
    return cache


if __name__ == '__main__':
    cache = build_cache()
    for ticker in ['AAPL', 'MSFT', 'NVDA']:
        if ticker in cache:
            c = cache[ticker]
            print(f"\n{ticker} ({c['name']}):")
            for k, v in c.items():
                if isinstance(v, float) and v is not None:
                    print(f"  {k:25s}: {v:.4f}")
                elif v is not None and str(v) != '':
                    print(f"  {k:25s}: {v}")
