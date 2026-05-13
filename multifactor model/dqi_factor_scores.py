"""
DQI Factor Scoring Engine — PROMETHEUS Parquet Edition
======================================================
Pipeline: Build cache → Winsorize → Sector-neutral z-score →
          Weighted composite → Percentile → Quintile rating

Identical scoring logic to DQI_version 2 / dqi_factor_scores.py.
Adapted to read from PROMETHEUS .parquet files via dqi_data_builder.

Usage:
    python dqi_factor_scores.py
    # Writes PROMETHEUS/Output/dqi_scores.parquet + dqi_scores.csv
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from dqi_config import (
    FACTOR_WEIGHTS, SUBFACTOR_WEIGHTS, INVERT_SUBFACTORS,
    WINSORIZE_PCT, MIN_WINSORIZE_SAMPLE, MIN_SECTOR_SIZE,
    MIN_COMPOSITE_COVERAGE, MAX_FACTOR_WEIGHT_MULT,
    RATING_BANDS, VIX_REGIME_THRESHOLDS, REGIME_FACTOR_WEIGHTS,
    OUTPUT_FILES,
)
from dqi_data_builder import build_cache


# ── Sub-factor → cache key mapping ───────────────────────────────────────────
# Maps the internal sub-factor name to the key used in the cache dict.
SUBFACTOR_FIELD_MAP = {
    # Value
    'ev_ebitda':       'ev_ebitda',
    'fcf_yield':       'fcf_yield',
    'earnings_yield':  'earnings_yield',
    # Quality
    'roic':            'roic',
    'net_debt_ebitda': 'net_debt_ebitda',
    'gross_margin':    'gross_margin',
    # Growth
    'rev_growth_ttm':  'rev_growth_ttm',
    'eps_growth_fy1':  'eps_growth_fy1',
    'rev_growth_ntm':  'rev_growth_ntm',
    # Momentum
    'return_12m_ex1':  'return_12m_ex1',
    'return_6m':       'return_6m',
    'return_12m':      'return_12m',
    # Revisions
    'eps_revision_1m': 'eps_revision_1m',
    # Profitability Momentum
    'delta_roic':         'delta_roic',
    'delta_gross_margin': 'delta_gross_margin',
    'delta_oper_margin':  'delta_oper_margin',
}

UNKNOWN_SECTOR = 'Unknown'


# ─── Pipeline functions (identical to DQI_version 2) ─────────────────────────

def winsorize(arr: np.ndarray,
              pct_low:  float = WINSORIZE_PCT[0],
              pct_high: float = WINSORIZE_PCT[1]) -> np.ndarray:
    """
    Winsorize a numpy array at pct_low/pct_high percentiles.
    Skips winsorization if fewer than MIN_WINSORIZE_SAMPLE valid values.
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_WINSORIZE_SAMPLE:
        return arr
    lo = np.nanpercentile(arr, pct_low  * 100)
    hi = np.nanpercentile(arr, pct_high * 100)
    return np.clip(arr, lo, hi)


def sector_neutral_zscore(values: np.ndarray, sectors: list,
                          min_sector_size: int = MIN_SECTOR_SIZE) -> np.ndarray:
    """
    Sector-neutral z-scores.
    Sectors with < min_sector_size valid observations fall back to universe stats.
    """
    z = np.full(len(values), np.nan)
    unique_sectors = {s for s in sectors if s and s != UNKNOWN_SECTOR}

    for sector in unique_sectors:
        idx = [i for i, s in enumerate(sectors) if s == sector]
        if len(idx) < min_sector_size:
            continue
        sect_vals  = values[idx]
        valid_mask = ~np.isnan(sect_vals)
        if valid_mask.sum() < min_sector_size:
            continue
        mean = np.nanmean(sect_vals)
        std  = np.nanstd(sect_vals)
        if std < 1e-10:
            z[idx] = 0.0
        else:
            for i in idx:
                if not np.isnan(values[i]):
                    z[i] = (values[i] - mean) / std

    # Fallback: any still-NaN ticker with valid data → universe stats
    no_z = np.isnan(z) & ~np.isnan(values)
    if no_z.sum() > 0:
        u_mean = np.nanmean(values)
        u_std  = np.nanstd(values)
        if u_std > 1e-10:
            for i in np.where(no_z)[0]:
                z[i] = (values[i] - u_mean) / u_std
        else:
            z[no_z] = 0.0

    return z


def assign_rating(percentile: float) -> str:
    """Map percentile rank (0–1) → rating string."""
    for threshold, rating in RATING_BANDS:
        if percentile >= threshold:
            return rating
    return 'POOR'


def detect_regime(latest_vix: float) -> str:
    """Return 'CALM' | 'NORMAL' | 'STRESS' based on latest VIX."""
    thresholds = VIX_REGIME_THRESHOLDS
    if latest_vix <= thresholds['CALM']:
        return 'CALM'
    if latest_vix <= thresholds['NORMAL']:
        return 'NORMAL'
    return 'STRESS'


def score_universe(cache: dict, factor_weights: dict = None) -> dict:
    """
    Score all tickers in the cache.

    Args:
        cache:          dqi_data_builder.build_cache() output
        factor_weights: override FACTOR_WEIGHTS (e.g. regime-switching)

    Returns:
        scores: {ticker: {composite_z, percentile, rating, factor_z,
                           subfactor_z, raw, sector, name, ...}}
    """
    if factor_weights is None:
        factor_weights = FACTOR_WEIGHTS

    tickers = list(cache.keys())
    n       = len(tickers)
    if n == 0:
        return {}

    sectors = [
        cache[t].get('gics_sub_industry_name') or UNKNOWN_SECTOR
        for t in tickers
    ]

    # ── Step 1: Extract & winsorize each sub-factor ──────────────────────────
    sub_factor_raw = {}
    for subfactor, cache_key in SUBFACTOR_FIELD_MAP.items():
        raw = np.array([
            float(cache[t][cache_key])
            if cache[t].get(cache_key) is not None
            else np.nan
            for t in tickers
        ], dtype=float)
        if subfactor in INVERT_SUBFACTORS:
            raw = -raw
        sub_factor_raw[subfactor] = winsorize(raw)

    # ── Step 2: Sector-neutral z-score per sub-factor ────────────────────────
    sub_factor_z = {
        sf: sector_neutral_zscore(raw, sectors)
        for sf, raw in sub_factor_raw.items()
    }

    # ── Step 3: Factor z-scores (weighted avg, NO re-z-scoring) ─────────────
    # Weighted average of sector-neutral z-scores is already comparable across
    # factors. Re-z-scoring would distort rankings by penalizing factors with
    # lower coverage (fix #4 from original engine).
    factor_z = {}
    for factor, subfactors in SUBFACTOR_WEIGHTS.items():
        composite   = np.zeros(n)
        weight_sum  = np.zeros(n)
        for sf, w in subfactors.items():
            z     = sub_factor_z.get(sf, np.full(n, np.nan))
            valid = ~np.isnan(z)
            composite[valid]  += w * z[valid]
            weight_sum[valid] += w
        f_z = np.full(n, np.nan)
        valid_mask = weight_sum > 0.10   # ≥10% weight coverage
        f_z[valid_mask] = composite[valid_mask] / weight_sum[valid_mask]
        factor_z[factor] = f_z

    # ── Step 4: Composite z-score with weight-drift cap ──────────────────────
    composite_z        = np.zeros(n)
    composite_w_sum    = np.zeros(n)
    for factor, w in factor_weights.items():
        fz    = factor_z.get(factor, np.full(n, np.nan))
        valid = ~np.isnan(fz)
        composite_z[valid]     += w * fz[valid]
        composite_w_sum[valid] += w

    composite_z_final = np.full(n, np.nan)
    for i in range(n):
        ws = composite_w_sum[i]
        if ws < MIN_COMPOSITE_COVERAGE:
            continue
        renorm = 1.0 / ws
        if renorm > MAX_FACTOR_WEIGHT_MULT:
            continue   # too much drift → mark N/A
        composite_z_final[i] = composite_z[i] / ws

    # ── Step 5: Percentile with midpoint ranking for ties ────────────────────
    valid_idx    = [i for i, v in enumerate(composite_z_final) if not np.isnan(v)]
    valid_scores = np.array([composite_z_final[i] for i in valid_idx])
    sorted_scores = np.sort(valid_scores)
    n_valid       = len(sorted_scores)

    def get_percentile(z: float) -> float:
        if np.isnan(z) or n_valid <= 1:
            return np.nan if np.isnan(z) else 0.5
        left     = int(np.searchsorted(sorted_scores, z, side='left'))
        right    = int(np.searchsorted(sorted_scores, z, side='right'))
        midpoint = (left + right - 1) / 2.0
        return midpoint / max(n_valid - 1, 1)

    # ── Step 6: Build output dict ─────────────────────────────────────────────
    scores = {}
    for i, ticker in enumerate(tickers):
        czf    = composite_z_final[i]
        pct    = get_percentile(czf)
        rating = assign_rating(pct) if not np.isnan(pct) else 'N/A'
        td     = cache[ticker]

        scores[ticker] = {
            'composite_z': float(czf) if not np.isnan(czf) else None,
            'percentile':  float(pct) if not np.isnan(pct) else None,
            'rating':      rating,
            'sector':      td.get('sector', UNKNOWN_SECTOR),
            'sub_industry':td.get('gics_sub_industry_name', UNKNOWN_SECTOR),
            'name':        td.get('name', ''),
            'currency':    td.get('currency', 'USD'),
            'px_last':     td.get('px_last'),
            'mkt_cap':     td.get('mkt_cap'),
            'analyst_recommendation': td.get('analyst_recommendation'),
            'factor_z': {
                f: (float(factor_z[f][i]) if not np.isnan(factor_z[f][i]) else None)
                for f in factor_z
            },
            'subfactor_z': {
                sf: (float(sub_factor_z[sf][i]) if not np.isnan(sub_factor_z[sf][i]) else None)
                for sf in sub_factor_z
            },
            'raw': {
                'ev_ebitda':       td.get('ev_ebitda'),
                'fcf_yield':       td.get('fcf_yield'),
                'earnings_yield':  td.get('earnings_yield'),
                'roic':            td.get('roic'),
                'net_debt_ebitda': td.get('net_debt_ebitda'),
                'gross_margin':    td.get('gross_margin'),
                'rev_growth_ttm':  td.get('rev_growth_ttm'),
                'eps_growth_fy1':  td.get('eps_growth_fy1'),
                'rev_growth_ntm':  td.get('rev_growth_ntm'),
                'return_12m_ex1':  td.get('return_12m_ex1'),
                'return_6m':       td.get('return_6m'),
                'return_12m':      td.get('return_12m'),
                'eps_revision_1m': td.get('eps_revision_1m'),
                'delta_roic':      td.get('delta_roic'),
                'delta_gm':        td.get('delta_gross_margin'),
                'delta_em':        td.get('delta_oper_margin'),
                'ttm_revenue':     td.get('ttm_revenue'),
                'ttm_net_income':  td.get('ttm_net_income'),
                'ttm_fcf':         td.get('ttm_fcf'),
                'mkt_cap':         td.get('mkt_cap'),
            },
        }

    return scores


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Full pipeline:
      1. Build cache from parquet files
      2. Detect VIX regime → select factor weights
      3. Score universe
      4. Print summary
      5. Save PROMETHEUS/Output/dqi_scores.parquet + .csv
    """
    # Build cache
    cache = build_cache(verbose=True)

    # Detect regime
    sample_vix   = next(iter(cache.values())).get('latest_vix', 20.0)
    regime       = detect_regime(sample_vix)
    regime_weights = REGIME_FACTOR_WEIGHTS[regime]
    print(f"\n  VIX={sample_vix:.1f} → Regime: {regime}")
    print(f"  Factor weights: {regime_weights}")

    # Score
    print("\nScoring universe...")
    scores = score_universe(cache, factor_weights=regime_weights)

    # Stats
    rated  = [s for s in scores.values() if s['rating'] != 'N/A']
    counts = Counter(s['rating'] for s in rated)
    print(f"  Scored: {len(rated):,} / {len(scores):,} tickers")
    print(f"  Ratings: {dict(counts)}")

    # Top 10 / Bottom 10
    valid = [(t, s) for t, s in scores.items() if s['composite_z'] is not None]
    valid.sort(key=lambda x: x[1]['composite_z'], reverse=True)

    print("\n  ── TOP 10 (STRONG) ──────────────────────────────────────")
    for t, s in valid[:10]:
        print(f"    {t:<10} {s['rating']:<8} z={s['composite_z']:+.3f}  {s['name'][:35]}")

    print("\n  ── BOTTOM 10 (POOR) ─────────────────────────────────────")
    for t, s in valid[-10:]:
        print(f"    {t:<10} {s['rating']:<8} z={s['composite_z']:+.3f}  {s['name'][:35]}")

    # ── Save output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILES['scores_parquet']), exist_ok=True)

    # Flatten to DataFrame
    rows = []
    for ticker, s in scores.items():
        row = {
            'symbol':      ticker,
            'name':        s['name'],
            'sector':      s['sector'],
            'sub_industry':s['sub_industry'],
            'currency':    s['currency'],
            'composite_z': s['composite_z'],
            'percentile':  s['percentile'],
            'rating':      s['rating'],
            'px_last':     s['px_last'],
            'mkt_cap':     s['mkt_cap'],
            'analyst_rec': s['analyst_recommendation'],
        }
        # Factor z-scores
        for f, fz in s['factor_z'].items():
            row[f'z_{f}'] = fz
        # Sub-factor z-scores
        for sf, sfz in s['subfactor_z'].items():
            row[f'zsf_{sf}'] = sfz
        # Raw values
        for k, v in s['raw'].items():
            row[f'raw_{k}'] = v
        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values('composite_z', ascending=False, na_position='last')
    df_out.to_parquet(OUTPUT_FILES['scores_parquet'], index=False)
    df_out.to_csv(OUTPUT_FILES['scores_csv'], index=False)

    print(f"\n  Saved → {OUTPUT_FILES['scores_parquet']}")
    print(f"  Saved → {OUTPUT_FILES['scores_csv']}")
    print(f"  Output shape: {df_out.shape}")

    return df_out


if __name__ == '__main__':
    main()
