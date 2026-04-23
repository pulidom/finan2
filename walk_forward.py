import numpy as np
import pandas as pd
from itertools import combinations

from cointegracion import (
    capital_invertido,
    invierte,
    invierte_con_estadisticas_fijas,
    zscore_moving_win,
)
from statistics import calculate_spread_off, stats
from read_data import load_ts, sector_d
import weights
import utils


class Conf:
    pathdat = "./dat/"
    tipo = "asset"
    Ntraining = 1008  # 4 anos de entrenamiento aproximados (~252 dias * 4)
    beta_win = 61
    zscore_win = 31
    sigma_co = 1.5
    sigma_ve = 0.1
    sigma_stop = 3.0

    # 1. Optimizacion de universo
    pre_screen_corr = 0.70
    pre_screen_top_n = 500
    nsel = 50
    pair_rank_stat_weight = 0.40
    pair_rank_sharpe_weight = 0.35
    pair_rank_drawdown_weight = 0.15
    pair_rank_trades_weight = 0.10
    pair_rank_trade_target = 10

    linver_betaweight = 0
    transaction_cost = 0.000
    forward_window = 126  # 6 meses exactos por ventana OOS
    max_extension = 5
    use_half_life_holding = False
    use_target_capital = False
    freeze_spread_stats_on_entry = False
    use_turnover_event_overlay = False
    turnover_z_window = 126
    turnover_z_threshold = 2.5
    turnover_penalty_lambda = 0.5
    oos_start = None
    oos_end = None

    # 2. Optimizacion de fechas
    init_date = "2010-01-01"
    end_date = "2017-12-31"

    # 3. Hiperparametros de weighting
    kelly_fraction = 1.0
    kelly_lookback = 20
    kelly_clip = 0.30
    kelly_target_capital = 1.0
    ggr_clip = 0.30
    ggr_target_capital = 1.0


def _signals_to_position(compras, ccompras):
    pos = np.zeros(len(compras))
    pos[np.asarray(compras, dtype=bool)] = 1.0
    pos[np.asarray(ccompras, dtype=bool)] = -1.0
    return pd.Series(pos).replace(0, np.nan).ffill().fillna(0).values


def _build_trading_signals(zscore, spread, spread_mean, spread_std, cnf):
    if getattr(cnf, "freeze_spread_stats_on_entry", False):
        compras, ccompras, zscore_eval = invierte_con_estadisticas_fijas(
            spread,
            spread_mean,
            spread_std,
            sigma_co=cnf.sigma_co,
            sigma_ve=cnf.sigma_ve,
            sigma_stop=cnf.sigma_stop,
        )
        return zscore_eval, compras, ccompras

    compras, ccompras = invierte(zscore, cnf.sigma_co, cnf.sigma_ve)
    return zscore, compras, ccompras


def _trade_diagnostics(position, returns, prev_position=0):
    position = np.asarray(position)
    returns = np.asarray(returns)

    durations = []
    trade_returns = []
    trade_records = []
    entries = 0
    carry_in_trades = 0
    open_trade_at_end = 0

    prev_sign = int(np.sign(prev_position))
    trade_start = None
    countable_trade = False

    if position.size > 0 and prev_sign != 0 and int(np.sign(position[0])) == prev_sign:
        trade_start = 0
        countable_trade = False
        carry_in_trades = 1

    for t, pos_t in enumerate(position):
        pos_sign = int(np.sign(pos_t))
        prior_sign = prev_sign if t == 0 else int(np.sign(position[t - 1]))

        if pos_sign != 0 and prior_sign == 0:
            trade_start = t
            countable_trade = True
            entries += 1
        elif pos_sign != 0 and prior_sign != 0 and pos_sign != prior_sign:
            if trade_start is not None and countable_trade:
                duration = t - trade_start
                trade_return = np.prod(1.0 + returns[trade_start:t]) - 1.0
                durations.append(duration)
                trade_returns.append(trade_return)
                trade_records.append(
                    {
                        "start": int(trade_start),
                        "end": int(t),
                        "duration": int(duration),
                        "return": float(trade_return),
                        "exit_type": "flip",
                    }
                )
            trade_start = t
            countable_trade = True
            entries += 1
        elif pos_sign == 0 and prior_sign != 0:
            if trade_start is not None and countable_trade:
                duration = t - trade_start
                trade_return = np.prod(1.0 + returns[trade_start:t]) - 1.0
                durations.append(duration)
                trade_returns.append(trade_return)
                trade_records.append(
                    {
                        "start": int(trade_start),
                        "end": int(t),
                        "duration": int(duration),
                        "return": float(trade_return),
                        "exit_type": "close",
                    }
                )
            trade_start = None
            countable_trade = False

    if position.size > 0 and position[-1] != 0 and trade_start is not None and countable_trade:
        duration = len(position) - trade_start
        trade_return = np.prod(1.0 + returns[trade_start:]) - 1.0
        durations.append(duration)
        trade_returns.append(trade_return)
        trade_records.append(
            {
                "start": int(trade_start),
                "end": int(len(position)),
                "duration": int(duration),
                "return": float(trade_return),
                "exit_type": "end",
            }
        )
        open_trade_at_end = 1

    return {
        "entries": entries,
        "durations": durations,
        "trade_returns": trade_returns,
        "trade_records": trade_records,
        "days_active_total": int(np.sum(position != 0)),
        "carry_in_trades": carry_in_trades,
        "open_trade_at_end": open_trade_at_end,
    }


def _activity_diagnostics(active_mask, prev_active=False):
    active_mask = np.asarray(active_mask, dtype=bool)
    durations = []
    entries = 0
    start_idx = None

    if active_mask.size > 0 and prev_active and active_mask[0]:
        start_idx = 0

    for t, is_active in enumerate(active_mask):
        prior_active = bool(prev_active) if t == 0 else bool(active_mask[t - 1])

        if is_active and not prior_active:
            entries += 1
            start_idx = t
        elif not is_active and prior_active and start_idx is not None:
            durations.append(int(t - start_idx))
            start_idx = None

    if active_mask.size > 0 and active_mask[-1] and start_idx is not None:
        durations.append(int(len(active_mask) - start_idx))

    return {
        "entries": int(entries),
        "durations": durations,
        "days_active_total": int(np.sum(active_mask)),
    }


def _weight_rebalance_cost(curr_signed_weights, prev_signed_weights, cost):
    """
    Cobra costo solo por cambios de peso en posiciones que permanecen abiertas
    con la misma dirección. Las aperturas/cierres/flips ya pagan costo dentro
    de capital_invertido a nivel señal del par.
    """
    if cost <= 0:
        return 0.0

    turnover = 0.0
    for pair_name, curr_w in curr_signed_weights.items():
        prev_w = prev_signed_weights.get(pair_name, 0.0)
        if abs(curr_w) <= weights.EPSILON or abs(prev_w) <= weights.EPSILON:
            continue
        if np.sign(curr_w) != np.sign(prev_w):
            continue
        turnover += abs(curr_w - prev_w)

    return float(cost * turnover)


def _calculate_metrics_from_returns(returns_array, initial_capital=100.0, risk_free_rate=0.02):
    returns_array = np.asarray(returns_array, dtype=float)
    if returns_array.size == 0:
        return {
            "cagr": 0.0,
            "max_dd": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "final_cap": float(initial_capital),
            "capital_curve": np.array([float(initial_capital)], dtype=float),
        }

    capital = float(initial_capital) * np.cumprod(1.0 + returns_array)
    final_cap = float(capital[-1])
    total_days = max(1, int(returns_array.size))
    cagr = (final_cap / float(initial_capital)) ** (252.0 / total_days) - 1.0
    peak = np.maximum.accumulate(capital)
    drawdown = (peak - capital) / np.maximum(peak, 1e-12)
    max_dd = float(np.max(drawdown)) if drawdown.size else 0.0
    vol = float(np.std(returns_array) * np.sqrt(252.0))
    sharpe = float((cagr - risk_free_rate) / vol) if vol > 0 else 0.0

    return {
        "cagr": float(cagr),
        "max_dd": max_dd,
        "vol": vol,
        "sharpe": sharpe,
        "final_cap": final_cap,
        "capital_curve": capital,
    }


def _rank_quality(values, higher_is_better=True):
    values = np.asarray(values, dtype=float)
    quality = np.full(values.shape[0], 0.5, dtype=float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) <= 1:
        return quality

    ranked_values = values[finite] if higher_is_better else -values[finite]
    if np.nanmax(ranked_values) - np.nanmin(ranked_values) <= 1e-12:
        return quality

    ranks = pd.Series(ranked_values).rank(method="average").to_numpy(dtype=float)
    quality[finite] = (ranks - 1.0) / max(len(ranks) - 1.0, 1.0)
    return quality


def _trade_count_quality(trade_counts, target_trades):
    target = max(float(target_trades), 1.0)
    counts = np.maximum(np.asarray(trade_counts, dtype=float), 0.0)
    return np.clip(np.sqrt(counts / target), 0.0, 1.0)


def _build_strategy_weights(
    positions,
    z_matrix,
    returns_matrix,
    spread_vols,
    crossings_rate,
    past_returns,
    selected_hl,
    pair_adverse_event_scores,
    tope_optimo,
    cnf,
):
    positions = np.asarray(positions, dtype=float)
    z_matrix = np.asarray(z_matrix, dtype=float)
    returns_matrix = np.asarray(returns_matrix, dtype=float)

    def finalize_weights(raw_weights):
        adjusted_weights = np.asarray(raw_weights, dtype=float)
        if getattr(cnf, "use_turnover_event_overlay", False):
            adjusted_weights = weights.apply_event_shock_overlay(
                adjusted_weights,
                pair_adverse_event_scores,
                penalty_lambda=cnf.turnover_penalty_lambda,
            )
        capped_weights = np.minimum(adjusted_weights, tope_optimo)
        if cnf.use_half_life_holding:
            return weights.apply_hold_period(
                capped_weights,
                positions,
                selected_hl,
                max_total=1.0,
            )
        return capped_weights

    return [
        ("Equal", finalize_weights(weights.equal_weight(positions))),
        ("Risk", finalize_weights(weights.risk_parity_weight(positions, spread_vols))),
        ("InvVar", finalize_weights(weights.inverse_variance_weight(positions, spread_vols))),
        ("ZScore", finalize_weights(weights.zscore_pure_weight(positions, z_matrix))),
        ("ZSquashed", finalize_weights(weights.zscore_squashed_weight(positions, z_matrix, spread_vols))),
        ("ZCross", finalize_weights(weights.crossings_weight(positions, z_matrix, spread_vols, crossings_rate))),
        (
            "GGR",
            finalize_weights(
                weights.ggr_weight(
                    positions,
                    z_matrix,
                    spread_vols,
                    target_capital=cnf.ggr_target_capital,
                    clip=cnf.ggr_clip,
                )
            ),
        ),
        (
            "Kelly",
            finalize_weights(
                weights.kelly_dynamic_weight(
                    positions,
                    z_matrix,
                    spread_vols,
                    kelly_fraction=cnf.kelly_fraction,
                    clip=cnf.kelly_clip,
                )
            ),
        ),
        (
            "KellyOU",
            finalize_weights(
                weights.kelly_ou_weight(
                    positions,
                    z_matrix,
                    returns_matrix,
                    spread_vols,
                    lookback=cnf.kelly_lookback,
                    kelly_fraction=cnf.kelly_fraction,
                    clip=cnf.kelly_clip,
                    target_capital=cnf.kelly_target_capital,
                )
            ),
        ),
        ("Inverse", finalize_weights(weights.zscore_inverse_weight(positions, z_matrix))),
        ("PastPerf", finalize_weights(weights.past_performance_weight(positions, past_returns))),
    ]


def _compute_strategy_returns(weight_matrix, returns_matrix, positions, pair_names, cost, prev_signed_weights=None):
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    positions = np.asarray(positions, dtype=float)
    signed_w = weight_matrix * np.sign(positions)
    n_days = returns_matrix.shape[1]
    strategy_rets = np.zeros(n_days, dtype=float)
    prev_signed = dict(prev_signed_weights or {})

    for t in range(n_days):
        curr_signed = {
            par_name: float(signed_w[p_idx, t])
            for p_idx, par_name in enumerate(pair_names)
            if abs(signed_w[p_idx, t]) > weights.EPSILON
        }
        rebalance_cost = _weight_rebalance_cost(curr_signed, prev_signed, cost)
        strategy_rets[t] = float(np.sum(weight_matrix[:, t] * returns_matrix[:, t])) - rebalance_cost
        prev_signed = curr_signed

    return strategy_rets, signed_w, prev_signed


def _compute_pair_adverse_event_scores(positions_fwd, selected_pairs, signed_event_scores, start_offset, end_offset):
    """
    Para cada par activo construye el score adverso:

        a_ij,t = [-q_ij,t s_i,t]_+ + [q_ij,t s_j,t]_+

    donde q_ij,t es la dirección de la posición del par y s_i,t es el score
    firmado del activo i derivado de sign(ret) * max(zturn, 0).
    """
    positions_fwd = np.asarray(positions_fwd, dtype=float)
    signed_event_scores = np.asarray(signed_event_scores, dtype=float)

    n_pairs, final_forward_len = positions_fwd.shape
    adverse_scores = np.zeros((n_pairs, final_forward_len), dtype=float)
    aligned_scores = signed_event_scores[:, start_offset:end_offset]

    if aligned_scores.shape[1] < final_forward_len:
        pad_width = final_forward_len - aligned_scores.shape[1]
        aligned_scores = np.pad(aligned_scores, ((0, 0), (0, pad_width)), mode="constant")
    elif aligned_scores.shape[1] > final_forward_len:
        aligned_scores = aligned_scores[:, :final_forward_len]

    for p_idx, (idx_i, idx_j) in enumerate(selected_pairs):
        q = np.sign(positions_fwd[p_idx, :])
        s_i = aligned_scores[idx_i, :]
        s_j = aligned_scores[idx_j, :]
        adverse = np.maximum(-q * s_i, 0.0) + np.maximum(q * s_j, 0.0)
        adverse[q == 0] = 0.0
        adverse_scores[p_idx, :] = adverse

    return adverse_scores


def run_walk_forward(sector="oil", cnf=None):
    if cnf is None:
        cnf = Conf()

    print(f"--- Iniciando Walk-Forward para {sector.upper()} ---")
    day, date, price, company, volume = load_ts(
        sectors=[sector],
        pathdat=cnf.pathdat,
        init_date=cnf.init_date,
        end_date=cnf.end_date,
    )
    t_start = cnf.Ntraining
    local_dates = pd.DatetimeIndex(pd.to_datetime(date))
    if getattr(cnf, "oos_start", None) is not None or getattr(cnf, "oos_end", None) is not None:
        oos_start = pd.Timestamp(getattr(cnf, "oos_start", local_dates.min()))
        oos_end = pd.Timestamp(getattr(cnf, "oos_end", local_dates.max()))
        oos_mask = (local_dates >= oos_start) & (local_dates <= oos_end)
        if not np.any(oos_mask):
            raise ValueError(
                f"{sector} no tiene observaciones dentro del rango OOS "
                f"{oos_start.date()} a {oos_end.date()}"
            )
        effective_oos_start = local_dates[oos_mask][0]
        pre_mask = local_dates < effective_oos_start
        mask = pre_mask | oos_mask
        date = local_dates[mask].to_pydatetime()
        price = price[:, mask]
        volume = volume[:, mask]
        day = np.arange(len(date))
        t_start = int(np.sum(pre_mask))

    if t_start < cnf.Ntraining:
        raise ValueError(
            f"{sector} no tiene suficientes observaciones previas para training: "
            f"{t_start} < {cnf.Ntraining}"
        )

    total_days = price.shape[1]

    strategy_names = [
        "Equal",
        "Risk",
        "InvVar",
        "ZScore",
        "ZSquashed",
        "ZCross",
        "GGR",
        "Kelly",
        "KellyOU",
        "Inverse",
        "PastPerf",
    ]

    all_dates = date[t_start:]
    out_of_sample_returns = {k: np.zeros(len(all_dates)) for k in strategy_names}
    out_of_sample_exposure = {k: np.zeros(len(all_dates)) for k in strategy_names}

    current_time_idx = t_start

    global_trades = 0
    global_trade_durations = []
    global_trade_records = []
    global_active_pairs = 0
    global_windows = 0
    global_pair_stats = {}
    global_active_pairs_timeline = np.zeros(len(all_dates), dtype=int)
    global_pre_filtered_counts = []
    global_strict_filtered_counts = []
    global_selected_counts = []
    global_selected_pairs_unique = set()
    selection_windows = []
    strategy_selection_windows = []
    prev_selected_pair_names = []
    prev_strict_pair_names = []
    strategy_activity_stats = {
        name: {"total_trades": 0, "durations": [], "days_active_total": 0}
        for name in strategy_names
    }
    strategy_active_pairs_timeline = {
        name: np.zeros(len(all_dates), dtype=int) for name in strategy_names
    }
    strategy_pair_weight_timeline = {name: {} for name in strategy_names}
    strategy_trade_records = {name: [] for name in strategy_names}
    strategy_prev_signed_weights = {name: {} for name in strategy_names}

    while current_time_idx < total_days - 1:
        t_train_start = current_time_idx - cnf.Ntraining
        t_train_end = current_time_idx

        if t_train_start < 0:
            break

        print(f"Ventana operativa iniciando en {date[current_time_idx].date()}")
        price_train = price[:, t_train_start:t_train_end]

        assets_train_l = list(combinations(range(price.shape[0]), 2))
        valid_pairs = []
        for i, j in assets_train_l:
            if not (np.isnan(price_train[i]).any() or np.isnan(price_train[j]).any()):
                valid_pairs.append((i, j))

        if not valid_pairs:
            print("No valid pairs found in this window.")
            current_time_idx = min(current_time_idx + cnf.forward_window, total_days)
            continue

        valid_set = sorted({a for pair in valid_pairs for a in pair})
        valid_idx_map = {orig: new for new, orig in enumerate(valid_set)}
        price_sub = price_train[valid_set, :]
        corr_matrix = np.corrcoef(price_sub)

        pre_filtered = []
        for i, j in valid_pairs:
            ci = valid_idx_map[i]
            cj = valid_idx_map[j]
            if abs(corr_matrix[ci, cj]) >= cnf.pre_screen_corr:
                pre_filtered.append((i, j))

        if len(pre_filtered) > cnf.pre_screen_top_n:
            corr_vals = [abs(corr_matrix[valid_idx_map[i], valid_idx_map[j]]) for i, j in pre_filtered]
            top_indices = np.argsort(corr_vals)[::-1][:cnf.pre_screen_top_n]
            pre_filtered = [pre_filtered[k] for k in top_indices]

        print(f"Pares tras pre-filtro de correlacion (>={cnf.pre_screen_corr}): {len(pre_filtered)}")
        global_pre_filtered_counts.append(len(pre_filtered))

        if not pre_filtered:
            print("No pairs passed correlation pre-filter, skipping window.")
            current_time_idx = min(current_time_idx + cnf.forward_window, total_days)
            continue

        pairs_data = [[price_train[i], price_train[j]] for i, j in pre_filtered]
        metrics = stats(pairs_data, cnf.tipo)
        valid_pairs = pre_filtered

        filtered_pairs = []
        filtered_scores = []
        filtered_hl = []
        filtered_trade_metrics = []

        for idx in range(len(valid_pairs)):
            p_val = metrics.pvalue[idx]
            hurst = metrics.hurst[idx]
            hl = metrics.half_life[idx]
            score = metrics.score[idx]

            if p_val < 0.05 and hurst < 0.45 and 1 <= hl <= 30:
                i, j = valid_pairs[idx]
                x, y, nret_x_strict, nret_y_strict = utils.select_variables(
                    price_train[i],
                    price_train[j],
                    cnf.tipo,
                )

                spread, _ = calculate_spread_off(x, y)
                spread_clean = np.nan_to_num(spread, nan=0.0)
                zero_crossings = np.sum(np.diff(np.sign(spread_clean)) != 0)

                if zero_crossings >= 12:
                    zs_raw, b, s, sm, ss = zscore_moving_win(x, y, cnf.beta_win, cnf.zscore_win)
                    zs_signal, compras, ccompras = _build_trading_signals(zs_raw, s, sm, ss, cnf)
                    beta_is = b if cnf.linver_betaweight else None
                    _, _, cap_is, retorno_is = capital_invertido(
                        nret_x_strict,
                        nret_y_strict,
                        compras,
                        ccompras,
                        beta=beta_is,
                        cost=cnf.transaction_cost,
                    )
                    filtered_pairs.append(valid_pairs[idx])
                    filtered_scores.append(score)
                    filtered_hl.append(hl)
                    pos_is = _signals_to_position(compras, ccompras)
                    diag_is = _trade_diagnostics(pos_is, retorno_is)
                    metrics_is = _calculate_metrics_from_returns(retorno_is)
                    filtered_trade_metrics.append(
                        {
                            "sharpe": float(metrics_is["sharpe"]),
                            "max_dd": float(metrics_is["max_dd"]),
                            "cagr": float(metrics_is["cagr"]),
                            "final_cap": float(cap_is[-1]) if len(cap_is) else 100.0,
                            "trades": int(diag_is["entries"]),
                        }
                    )

        if not filtered_pairs:
            print("No pairs survived the strict filters in this window.")
            current_time_idx = min(current_time_idx + cnf.forward_window, total_days)
            continue

        print(f"Pares que sobrevivieron a los filtros estrictos: {len(filtered_pairs)}")
        global_strict_filtered_counts.append(len(filtered_pairs))

        strict_pair_names = [f"{company[i]} - {company[j]}" for i, j in filtered_pairs]
        stat_quality = _rank_quality(filtered_scores, higher_is_better=False)
        sharpe_quality = _rank_quality(
            [item["sharpe"] for item in filtered_trade_metrics],
            higher_is_better=True,
        )
        drawdown_quality = _rank_quality(
            [item["max_dd"] for item in filtered_trade_metrics],
            higher_is_better=False,
        )
        trades_quality = _trade_count_quality(
            [item["trades"] for item in filtered_trade_metrics],
            cnf.pair_rank_trade_target,
        )
        combined_pair_quality = (
            cnf.pair_rank_stat_weight * stat_quality
            + cnf.pair_rank_sharpe_weight * sharpe_quality
            + cnf.pair_rank_drawdown_weight * drawdown_quality
            + cnf.pair_rank_trades_weight * trades_quality
        )
        idx_sorted = sorted(
            range(len(filtered_pairs)),
            key=lambda k: (
                -combined_pair_quality[k],
                filtered_scores[k],
                -filtered_trade_metrics[k]["sharpe"],
                filtered_trade_metrics[k]["max_dd"],
                -filtered_trade_metrics[k]["trades"],
            ),
        )[: cnf.nsel]
        selected_pairs = [filtered_pairs[k] for k in idx_sorted]
        selected_pair_names = [f"{company[i]} - {company[j]}" for i, j in selected_pairs]
        selected_hl = np.array([filtered_hl[k] for k in idx_sorted], dtype=float)
        selected_pair_ranking = []
        for rank_idx, k in enumerate(idx_sorted, start=1):
            metrics_k = filtered_trade_metrics[k]
            selected_pair_ranking.append(
                {
                    "rank": int(rank_idx),
                    "pair": strict_pair_names[k],
                    "combined_quality": float(combined_pair_quality[k]),
                    "stat_score": float(filtered_scores[k]),
                    "stat_quality": float(stat_quality[k]),
                    "sharpe": float(metrics_k["sharpe"]),
                    "sharpe_quality": float(sharpe_quality[k]),
                    "max_dd": float(metrics_k["max_dd"]),
                    "drawdown_quality": float(drawdown_quality[k]),
                    "trades": int(metrics_k["trades"]),
                    "trades_quality": float(trades_quality[k]),
                    "cagr": float(metrics_k["cagr"]),
                    "final_cap": float(metrics_k["final_cap"]),
                }
            )
        global_selected_counts.append(len(selected_pairs))
        for i, j in selected_pairs:
            global_selected_pairs_unique.add(tuple(sorted((str(company[i]), str(company[j])))))

        past_returns = np.zeros(len(selected_pairs))
        spread_vols_train = np.zeros(len(selected_pairs))
        crossings_rate_train = np.zeros(len(selected_pairs))
        train_alphas = np.zeros(len(selected_pairs))
        train_betas = np.zeros(len(selected_pairs))
        positions_insample = np.zeros((len(selected_pairs), cnf.Ntraining))
        zmatrix_insample = np.zeros((len(selected_pairs), cnf.Ntraining))
        returns_insample = np.zeros((len(selected_pairs), cnf.Ntraining))
        volume_train = volume[:, t_train_start:t_train_end]

        for p_idx, (i, j) in enumerate(selected_pairs):
            x, y, nret_x, nret_y = utils.select_variables(price_train[i], price_train[j], cnf.tipo)

            slope, intercept = utils.lin_reg(x, y)
            train_betas[p_idx] = slope
            train_alphas[p_idx] = intercept

            zs_raw, b, s, sm, ss = zscore_moving_win(x, y, cnf.beta_win, cnf.zscore_win)
            zs_signal, compras, ccompras = _build_trading_signals(zs_raw, s, sm, ss, cnf)

            beta_is = b if cnf.linver_betaweight else None
            _, _, cap, retorno_is = capital_invertido(
                nret_x,
                nret_y,
                compras,
                ccompras,
                beta=beta_is,
                cost=cnf.transaction_cost,
            )
            past_returns[p_idx] = (cap[-1] / cap[0]) - 1
            pos_is = _signals_to_position(compras, ccompras)
            positions_insample[p_idx, : min(cnf.Ntraining, len(pos_is))] = pos_is[:cnf.Ntraining]
            zs_is = np.nan_to_num(zs_signal, nan=0.0)
            zmatrix_insample[p_idx, : min(cnf.Ntraining, len(zs_is))] = zs_is[:cnf.Ntraining]
            returns_insample[p_idx, : min(cnf.Ntraining, len(retorno_is))] = retorno_is[:cnf.Ntraining]

            s_clean = np.nan_to_num(s, nan=0.0)
            spread_vols_train[p_idx] = np.nanstd(s)
            zero_crossings = np.where(np.diff(np.sign(s_clean)))[0]
            crossings_rate_train[p_idx] = len(zero_crossings) / float(len(s))

        if cnf.use_target_capital:
            tope_optimo, _ = weights.calcular_tope_por_promedio_señales(positions_insample)
        else:
            tope_optimo = 1.0

        t_forward_target = min(current_time_idx + cnf.forward_window, total_days)
        turnover_lookback = cnf.turnover_z_window if getattr(cnf, "use_turnover_event_overlay", False) else 0
        lookback = max(cnf.beta_win, cnf.zscore_win, turnover_lookback) + 1
        t_data_start = current_time_idx - lookback
        t_data_end = min(t_forward_target + 1, total_days)
        forward_len = t_forward_target - current_time_idx

        if forward_len <= 1:
            break

        all_compras_raw = []
        all_ccompras_raw = []
        base_slices = []
        signed_event_scores = None

        if getattr(cnf, "use_turnover_event_overlay", False):
            signed_event_scores = weights.signed_turnover_event_scores(
                price[:, t_data_start:t_data_end],
                volume[:, t_data_start:t_data_end],
                window=cnf.turnover_z_window,
                z_threshold=cnf.turnover_z_threshold,
            )

        for p_idx, (idx_i, idx_j) in enumerate(selected_pairs):
            x = price[idx_i, t_data_start:t_data_end]
            y = price[idx_j, t_data_start:t_data_end]

            x_fwd, y_fwd, nret_x_fwd, nret_y_fwd = utils.select_variables(x, y, cnf.tipo)

            import cointegracion

            zs_raw, b, spr, sm, ss = cointegracion.zscore_fixed_beta(
                x_fwd,
                y_fwd,
                train_alphas[p_idx],
                train_betas[p_idx],
                cnf.zscore_win,
            )
            zs_signal, compras, ccompras = _build_trading_signals(zs_raw, spr, sm, ss, cnf)

            all_compras_raw.append(compras)
            all_ccompras_raw.append(ccompras)
            base_slices.append((nret_x_fwd, nret_y_fwd, b, spr, zs_signal))

        all_compras_raw = np.array(all_compras_raw)
        all_ccompras_raw = np.array(all_ccompras_raw)

        actual_end_offset = lookback + forward_len
        final_forward_len = forward_len
        if final_forward_len <= 1:
            break

        positions_fwd = np.zeros((len(selected_pairs), final_forward_len))
        zmatrix_fwd = np.zeros((len(selected_pairs), final_forward_len))
        returns_fwd = np.zeros((len(selected_pairs), final_forward_len))
        prev_positions_fwd = np.zeros(len(selected_pairs), dtype=int)

        for p_idx in range(len(selected_pairs)):
            nret_x_fwd, nret_y_fwd, b, spr, zs_signal = base_slices[p_idx]

            compras_slice = all_compras_raw[p_idx, :actual_end_offset].copy()
            ccompras_slice = all_ccompras_raw[p_idx, :actual_end_offset].copy()

            if len(compras_slice) > 0 and compras_slice[-1]:
                compras_slice[-1] = False
            if len(ccompras_slice) > 0 and ccompras_slice[-1]:
                ccompras_slice[-1] = False

            beta_slice = b[:actual_end_offset] if cnf.linver_betaweight else None

            _, _, _, rets_array = capital_invertido(
                nret_x_fwd[: actual_end_offset - 1],
                nret_y_fwd[: actual_end_offset - 1],
                compras_slice,
                ccompras_slice,
                beta=beta_slice,
                cost=cnf.transaction_cost,
            )

            prev_idx = lookback - 2
            if prev_idx >= 0:
                if compras_slice[prev_idx]:
                    prev_positions_fwd[p_idx] = 1
                elif ccompras_slice[prev_idx]:
                    prev_positions_fwd[p_idx] = -1

            fwd_compras = compras_slice[lookback - 1 : actual_end_offset - 1]
            fwd_ccompras = ccompras_slice[lookback - 1 : actual_end_offset - 1]
            positions_fwd[p_idx, :] = _signals_to_position(fwd_compras, fwd_ccompras)

            fwd_rets = rets_array[lookback:actual_end_offset]
            if len(fwd_rets) < final_forward_len:
                fwd_rets = np.pad(fwd_rets, (0, final_forward_len - len(fwd_rets)), "constant")
            elif len(fwd_rets) > final_forward_len:
                fwd_rets = fwd_rets[:final_forward_len]
            returns_fwd[p_idx, :] = fwd_rets

            fwd_zs = np.nan_to_num(zs_signal[lookback - 1 : actual_end_offset - 1], nan=0.0)
            if len(fwd_zs) < final_forward_len:
                fwd_zs = np.pad(fwd_zs, (0, final_forward_len - len(fwd_zs)), "constant")
            elif len(fwd_zs) > final_forward_len:
                fwd_zs = fwd_zs[:final_forward_len]
            zmatrix_fwd[p_idx, :] = fwd_zs

        print(f" -> Bloque procesado (Longitud: {final_forward_len} dias)")

        out_idx_start = current_time_idx - t_start
        out_idx_end = out_idx_start + final_forward_len

        if out_idx_end > len(all_dates):
            clip = len(all_dates) - out_idx_start
            positions_fwd = positions_fwd[:, :clip]
            zmatrix_fwd = zmatrix_fwd[:, :clip]
            returns_fwd = returns_fwd[:, :clip]
            final_forward_len = clip
            out_idx_end = len(all_dates)

        if final_forward_len <= 0:
            break

        if getattr(cnf, "use_turnover_event_overlay", False):
            signed_event_scores_train = weights.signed_turnover_event_scores(
                price_train,
                volume_train,
                window=cnf.turnover_z_window,
                z_threshold=cnf.turnover_z_threshold,
            )
            pair_adverse_event_scores_train = _compute_pair_adverse_event_scores(
                positions_insample,
                selected_pairs,
                signed_event_scores_train,
                0,
                positions_insample.shape[1],
            )
        else:
            pair_adverse_event_scores_train = np.zeros_like(positions_insample, dtype=float)

        if signed_event_scores is not None:
            pair_adverse_event_scores = _compute_pair_adverse_event_scores(
                positions_fwd,
                selected_pairs,
                signed_event_scores,
                lookback - 1,
                actual_end_offset - 1,
            )
        else:
            pair_adverse_event_scores = np.zeros_like(positions_fwd, dtype=float)

        for p_idx, (idx_i, idx_j) in enumerate(selected_pairs):
            par_name = f"{company[idx_i]} - {company[idx_j]}"
            diag = _trade_diagnostics(
                positions_fwd[p_idx, :],
                returns_fwd[p_idx, :],
                prev_position=prev_positions_fwd[p_idx],
            )

            global_trades += diag["entries"]
            global_trade_durations.extend(diag["durations"])

            if par_name not in global_pair_stats:
                global_pair_stats[par_name] = {
                    "trades": 0,
                    "duracion": 0.0,
                    "durations": [],
                    "trade_returns": [],
                    "retorno_acumulado": 1.0,
                    "days_active_total": 0,
                    "carry_in_trades": 0,
                    "windows_selected": 0,
                }

            pair_stats = global_pair_stats[par_name]
            pair_stats["trades"] += diag["entries"]
            pair_stats["duracion"] += sum(diag["durations"])
            pair_stats["durations"].extend(diag["durations"])
            pair_stats["trade_returns"].extend(diag["trade_returns"])
            pair_stats["days_active_total"] += diag["days_active_total"]
            pair_stats["carry_in_trades"] += diag["carry_in_trades"]
            pair_stats["windows_selected"] += 1
            pair_stats["retorno_acumulado"] *= np.prod(1.0 + returns_fwd[p_idx, :])

            for trade_record in diag["trade_records"]:
                start_idx = out_idx_start + trade_record["start"]
                end_idx_exclusive = min(out_idx_start + trade_record["end"], len(all_dates))
                if start_idx >= len(all_dates) or end_idx_exclusive <= start_idx:
                    continue

                local_start_idx = int(trade_record["start"])
                local_last_active_idx = max(0, min(int(trade_record["end"]) - 1, final_forward_len - 1))
                local_exit_signal_idx = max(0, min(int(trade_record["end"]), final_forward_len - 1))
                exit_signal_idx = out_idx_start + local_exit_signal_idx
                exit_reason = str(trade_record.get("exit_type", "end"))
                global_trade_records.append(
                    {
                        "pair": par_name,
                        "sector": sector,
                        "start_index": int(start_idx),
                        "end_index": int(end_idx_exclusive),
                        "exit_signal_index": int(exit_signal_idx),
                        "start_date": pd.Timestamp(all_dates[start_idx]).isoformat(),
                        "end_date": pd.Timestamp(all_dates[end_idx_exclusive - 1]).isoformat(),
                        "exit_signal_date": pd.Timestamp(all_dates[exit_signal_idx]).isoformat(),
                        "duration": int(trade_record["duration"]),
                        "return": float(trade_record["return"]),
                        "half_life": float(selected_hl[p_idx]),
                        "side": int(np.sign(positions_fwd[p_idx, local_start_idx])),
                        "entry_zscore": float(zmatrix_fwd[p_idx, local_start_idx]),
                        "last_active_zscore": float(zmatrix_fwd[p_idx, local_last_active_idx]),
                        "exit_zscore": float(zmatrix_fwd[p_idx, local_exit_signal_idx]),
                        "exit_reason": exit_reason,
                    }
                )

        global_active_pairs += len(selected_pairs)
        global_windows += 1
        global_active_pairs_timeline[out_idx_start:out_idx_end] = np.count_nonzero(positions_fwd, axis=0)
        overlap_prev_strict_pairs = sorted(set(prev_strict_pair_names).intersection(strict_pair_names))
        overlap_prev_pairs = sorted(set(prev_selected_pair_names).intersection(selected_pair_names))
        strategies_insample = _build_strategy_weights(
            positions_insample,
            zmatrix_insample,
            returns_insample,
            spread_vols_train,
            crossings_rate_train,
            past_returns,
            selected_hl,
            pair_adverse_event_scores_train,
            tope_optimo,
            cnf,
        )
        strategy_train_metrics = {}
        for name, w_is in strategies_insample:
            strategy_rets_is, _, _ = _compute_strategy_returns(
                w_is,
                returns_insample,
                positions_insample,
                selected_pair_names,
                cnf.transaction_cost,
            )
            metrics_is = _calculate_metrics_from_returns(strategy_rets_is)
            strategy_train_metrics[name] = {
                "cagr": float(metrics_is["cagr"]),
                "sharpe": float(metrics_is["sharpe"]),
                "max_dd": float(metrics_is["max_dd"]),
                "final_cap": float(metrics_is["final_cap"]),
            }

        best_training_strategy = max(
            strategy_train_metrics.items(),
            key=lambda item: (item[1]["cagr"], item[1]["final_cap"], item[1]["sharpe"]),
        )[0]

        selection_windows.append(
            {
                "window_index": int(len(selection_windows) + 1),
                "training_start_date": pd.Timestamp(date[t_train_start]).isoformat(),
                "training_end_date": pd.Timestamp(date[t_train_end - 1]).isoformat(),
                "operation_start_date": pd.Timestamp(all_dates[out_idx_start]).isoformat(),
                "operation_end_date": pd.Timestamp(all_dates[out_idx_end - 1]).isoformat(),
                "training_days": int(cnf.Ntraining),
                "forward_days": int(final_forward_len),
                "valid_pairs": int(len(valid_pairs)),
                "pre_filtered_pairs": int(len(pre_filtered)),
                "strict_filtered_pairs": int(len(filtered_pairs)),
                "strict_overlap_prev_count": int(len(overlap_prev_strict_pairs)),
                "selected_pairs_count": int(len(selected_pair_names)),
                "overlap_prev_count": int(len(overlap_prev_pairs)),
                "strict_pairs": strict_pair_names,
                "strict_overlap_prev_pairs": overlap_prev_strict_pairs,
                "selected_pairs": selected_pair_names,
                "selected_pair_ranking": selected_pair_ranking,
                "overlap_prev_pairs": overlap_prev_pairs,
                "pair_rank_weights": {
                    "stat": float(cnf.pair_rank_stat_weight),
                    "sharpe": float(cnf.pair_rank_sharpe_weight),
                    "drawdown": float(cnf.pair_rank_drawdown_weight),
                    "trades": float(cnf.pair_rank_trades_weight),
                    "trade_target": float(cnf.pair_rank_trade_target),
                },
                "strategy_train_metrics": strategy_train_metrics,
                "best_training_strategy": best_training_strategy,
                "out_idx_start": int(out_idx_start),
                "out_idx_end": int(out_idx_end),
            }
        )
        strategy_selection_windows.append(
            {
                "window_index": int(len(strategy_selection_windows) + 1),
                "training_start_date": pd.Timestamp(date[t_train_start]).isoformat(),
                "training_end_date": pd.Timestamp(date[t_train_end - 1]).isoformat(),
                "operation_start_date": pd.Timestamp(all_dates[out_idx_start]).isoformat(),
                "operation_end_date": pd.Timestamp(all_dates[out_idx_end - 1]).isoformat(),
                "best_training_strategy": best_training_strategy,
                "strategy_train_metrics": strategy_train_metrics,
                "out_idx_start": int(out_idx_start),
                "out_idx_end": int(out_idx_end),
            }
        )
        prev_strict_pair_names = list(strict_pair_names)
        prev_selected_pair_names = list(selected_pair_names)

        strategies = _build_strategy_weights(
            positions_fwd,
            zmatrix_fwd,
            returns_fwd,
            spread_vols_train,
            crossings_rate_train,
            past_returns,
            selected_hl,
            pair_adverse_event_scores,
            tope_optimo,
            cnf,
        )

        for name, w in strategies:
            prev_signed_at_block_start = dict(strategy_prev_signed_weights[name])
            strategy_rets, signed_w, prev_signed = _compute_strategy_returns(
                w,
                returns_fwd,
                positions_fwd,
                selected_pair_names,
                cnf.transaction_cost,
                prev_signed_weights=strategy_prev_signed_weights[name],
            )
            strategy_prev_signed_weights[name] = prev_signed
            out_of_sample_returns[name][out_idx_start:out_idx_end] = strategy_rets
            out_of_sample_exposure[name][out_idx_start:out_idx_end] = np.sum(np.abs(w), axis=0)
            active_mask = np.abs(w) > weights.EPSILON
            strategy_active_pairs_timeline[name][out_idx_start:out_idx_end] = np.count_nonzero(active_mask, axis=0)
            for p_idx, par_name in enumerate(selected_pair_names):
                if par_name not in strategy_pair_weight_timeline[name]:
                    strategy_pair_weight_timeline[name][par_name] = np.zeros(len(all_dates), dtype=float)
                strategy_pair_weight_timeline[name][par_name][out_idx_start:out_idx_end] = signed_w[p_idx, :]

                effective_signed_position = np.where(
                    np.abs(w[p_idx, :]) > weights.EPSILON,
                    np.sign(positions_fwd[p_idx, :]),
                    0.0,
                )
                prev_position = int(np.sign(prev_signed_at_block_start.get(par_name, 0.0)))
                diag = _trade_diagnostics(
                    effective_signed_position,
                    returns_fwd[p_idx, :],
                    prev_position=prev_position,
                )
                strategy_activity_stats[name]["total_trades"] += diag["entries"]
                strategy_activity_stats[name]["durations"].extend(diag["durations"])
                strategy_activity_stats[name]["days_active_total"] += diag["days_active_total"]
                for trade_record in diag["trade_records"]:
                    start_idx = out_idx_start + trade_record["start"]
                    end_idx_exclusive = min(out_idx_start + trade_record["end"], len(all_dates))
                    if start_idx >= len(all_dates) or end_idx_exclusive <= start_idx:
                        continue

                    local_start_idx = int(trade_record["start"])
                    local_last_active_idx = max(0, min(int(trade_record["end"]) - 1, final_forward_len - 1))
                    local_exit_signal_idx = max(0, min(int(trade_record["end"]), final_forward_len - 1))
                    exit_signal_idx = out_idx_start + local_exit_signal_idx
                    exit_reason = str(trade_record.get("exit_type", "end"))
                    strategy_trade_records[name].append(
                        {
                            "pair": par_name,
                            "sector": sector,
                            "start_index": int(start_idx),
                            "end_index": int(end_idx_exclusive),
                            "exit_signal_index": int(exit_signal_idx),
                            "start_date": pd.Timestamp(all_dates[start_idx]).isoformat(),
                            "end_date": pd.Timestamp(all_dates[end_idx_exclusive - 1]).isoformat(),
                            "exit_signal_date": pd.Timestamp(all_dates[exit_signal_idx]).isoformat(),
                            "duration": int(trade_record["duration"]),
                            "return": float(trade_record["return"]),
                            "half_life": float(selected_hl[p_idx]),
                            "side": int(np.sign(effective_signed_position[local_start_idx])),
                            "entry_zscore": float(zmatrix_fwd[p_idx, local_start_idx]),
                            "last_active_zscore": float(zmatrix_fwd[p_idx, local_last_active_idx]),
                            "exit_zscore": float(zmatrix_fwd[p_idx, local_exit_signal_idx]),
                            "exit_reason": exit_reason,
                        }
                    )

        current_time_idx = t_forward_target

    print(f"--- Fin Walk-Forward {sector.upper()} ---")

    pair_stats_summary = {}
    for par_name, data in global_pair_stats.items():
        durations = data.pop("durations")
        trade_returns = data.pop("trade_returns")

        pair_stats_summary[par_name] = {
            **data,
            "avg_holding_days": float(np.mean(durations)) if durations else 0.0,
            "median_holding_days": float(np.median(durations)) if durations else 0.0,
            "max_holding_days": int(np.max(durations)) if durations else 0,
            "min_holding_days": int(np.min(durations)) if durations else 0,
            "avg_trade_return": float(np.mean(trade_returns)) if trade_returns else 0.0,
        }

    stats_dict = {
        "total_trades": global_trades,
        "avg_trade_duration": float(np.mean(global_trade_durations)) if global_trade_durations else 0.0,
        "median_trade_duration": float(np.median(global_trade_durations)) if global_trade_durations else 0.0,
        "max_trade_duration": int(np.max(global_trade_durations)) if global_trade_durations else 0,
        "min_trade_duration": int(np.min(global_trade_durations)) if global_trade_durations else 0,
        "avg_pairs_per_window": global_active_pairs / max(1, global_windows) if global_windows > 0 else 0.0,
        "avg_pre_filtered_pairs": float(np.mean(global_pre_filtered_counts)) if global_pre_filtered_counts else 0.0,
        "avg_strict_filtered_pairs": float(np.mean(global_strict_filtered_counts)) if global_strict_filtered_counts else 0.0,
        "avg_selected_pairs": float(np.mean(global_selected_counts)) if global_selected_counts else 0.0,
        "unique_selected_pairs": int(len(global_selected_pairs_unique)),
    }

    strategy_stats_summary = {}
    for name, data in strategy_activity_stats.items():
        durations = data["durations"]
        strategy_stats_summary[name] = {
            "total_trades": int(data["total_trades"]),
            "avg_trade_duration": float(np.mean(durations)) if durations else 0.0,
            "median_trade_duration": float(np.median(durations)) if durations else 0.0,
            "max_trade_duration": int(np.max(durations)) if durations else 0,
            "min_trade_duration": int(np.min(durations)) if durations else 0,
            "days_active_total": int(data["days_active_total"]),
        }

    return {
        "sector": sector,
        "available_sectors": sorted(sector_d.keys()),
        "dates": all_dates,
        "returns": out_of_sample_returns,
        "exposure": out_of_sample_exposure,
        "stats": stats_dict,
        "strategy_stats": strategy_stats_summary,
        "trade_durations": [int(x) for x in global_trade_durations],
        "trade_records": global_trade_records,
        "strategy_trade_records": strategy_trade_records,
        "pair_stats": pair_stats_summary,
        "selection_windows": selection_windows,
        "strategy_selection_windows": strategy_selection_windows,
        "active_pairs_timeline": global_active_pairs_timeline.tolist(),
        "strategy_active_pairs_timeline": {
            name: timeline.tolist() for name, timeline in strategy_active_pairs_timeline.items()
        },
        "strategy_pair_weight_timeline": {
            name: {pair: series.tolist() for pair, series in per_pair.items()}
            for name, per_pair in strategy_pair_weight_timeline.items()
        },
        "cnf": {
            "Ntraining": cnf.Ntraining,
            "beta_win": cnf.beta_win,
            "zscore_win": cnf.zscore_win,
            "sigma_co": cnf.sigma_co,
            "sigma_ve": cnf.sigma_ve,
            "sigma_stop": cnf.sigma_stop,
            "nsel": cnf.nsel,
            "pair_rank_weights": {
                "stat": float(cnf.pair_rank_stat_weight),
                "sharpe": float(cnf.pair_rank_sharpe_weight),
                "drawdown": float(cnf.pair_rank_drawdown_weight),
                "trades": float(cnf.pair_rank_trades_weight),
                "trade_target": float(cnf.pair_rank_trade_target),
            },
            "transaction_cost": cnf.transaction_cost,
            "forward_window": cnf.forward_window,
            "use_half_life_holding": cnf.use_half_life_holding,
            "use_target_capital": cnf.use_target_capital,
            "freeze_spread_stats_on_entry": cnf.freeze_spread_stats_on_entry,
            "use_turnover_event_overlay": cnf.use_turnover_event_overlay,
            "turnover_z_window": cnf.turnover_z_window,
            "turnover_z_threshold": cnf.turnover_z_threshold,
            "turnover_penalty_lambda": cnf.turnover_penalty_lambda,
            "kelly_fraction": cnf.kelly_fraction,
            "kelly_lookback": cnf.kelly_lookback,
            "kelly_clip": cnf.kelly_clip,
            "kelly_target_capital": cnf.kelly_target_capital,
            "ggr_clip": cnf.ggr_clip,
            "ggr_target_capital": cnf.ggr_target_capital,
        },
    }


if __name__ == "__main__":
    res = run_walk_forward("oil")
    print("Done. Returns length:", len(res["dates"]))
