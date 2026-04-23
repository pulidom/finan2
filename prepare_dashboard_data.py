import argparse
import itertools
import os
import pickle
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from metrics import calculate_metrics
from walk_forward import Conf, run_walk_forward

try:
    import yfinance as yf
except Exception:  # pragma: no cover - fallback defensivo
    yf = None


STRATEGY_ORDER = [
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
INDUSTRY_RESULT_SCHEMA_VERSION = 21


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera el cache offline que consume app.py para la presentacion."
    )
    parser.add_argument("--pathdat", default="./dat/", help="Carpeta de datos.")
    parser.add_argument("--ranking-file", default="./dat/industry_ranking.csv", help="Ranking de industrias sugeridas.")
    parser.add_argument("--index-file", default="./dat/industry_npz_index.csv", help="Indice de npz por industria.")
    parser.add_argument(
        "--results-dir",
        default="./dat/industry_results",
        help="Carpeta donde se guarda un pickle de resultados por industria.",
    )
    parser.add_argument("--top-industries", type=int, default=10, help="Cuantas industrias tomar del ranking.")
    parser.add_argument(
        "--selection-start",
        default="2014-01-01",
        help="Parametro legado. En modo adaptativo ya no se usa un tramo de seleccion separado.",
    )
    parser.add_argument(
        "--selection-end",
        default="2015-12-31",
        help="Parametro legado. En modo adaptativo ya no se usa un tramo de seleccion separado.",
    )
    parser.add_argument("--operation-start", default="2016-01-01", help="Fecha inicial del periodo real de operacion.")
    parser.add_argument("--operation-end", default="2017-12-31", help="Fecha final del periodo real de operacion.")
    parser.add_argument(
        "--sectors",
        default="",
        help="Lista manual de industrias separadas por coma. Si se pasa, reemplaza al ranking.",
    )
    parser.add_argument(
        "--selection-metric",
        default="cagr",
        choices=["sharpe", "cagr", "capital_final", "max_drawdown"],
        help="Metrica usada para elegir la mejor estrategia on-sample por industria en cada bloque walk-forward.",
    )
    parser.add_argument(
        "--selection-ratio",
        type=float,
        default=1.0,
        help="Parametro legado. Con el protocolo de etapa 2 y etapa 3 ya no se usa para partir el OOS.",
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Ticker del benchmark a descargar para comparar la evolucion general. Usar vacio para desactivar.",
    )
    parser.add_argument(
        "--output",
        default="./dat/dashboard_cache.pkl",
        help="Archivo pickle que luego consume app.py.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignora resultados por industria ya calculados y vuelve a correr todo para las industrias seleccionadas.",
    )
    parser.add_argument(
        "--freeze-stop-z",
        type=float,
        default=3.0,
        help="Umbral de seguridad para cerrar posiciones con mu/sigma fijos si el z-score congelado sigue alejandose.",
    )
    return parser.parse_args()


def ordered_names(names):
    names = list(names)
    ordered = [name for name in STRATEGY_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in STRATEGY_ORDER))
    return ordered


def build_combo_grid():
    options = [True, False]
    # transaction_cost queda fijo en activo; la re-seleccion cada 126 dias queda siempre activa.
    return [(True, True) + combo for combo in itertools.product(options, repeat=4)]


def build_conf_from_combo(combo, freeze_stop_z=3.0):
    tc_bool, fw_bool, hl_bool, cap_bool, freeze_bool, shock_bool = combo
    cnf = Conf()
    cnf.transaction_cost = 0.005 if tc_bool else 0.0
    cnf.forward_window = 126
    cnf.use_half_life_holding = hl_bool
    cnf.use_target_capital = cap_bool
    cnf.freeze_spread_stats_on_entry = freeze_bool
    cnf.use_turnover_event_overlay = shock_bool
    cnf.sigma_stop = float(freeze_stop_z)
    return cnf


def combo_to_label(combo):
    tc_bool, fw_bool, hl_bool, cap_bool, freeze_bool, shock_bool = combo
    return (
        f"tc={int(tc_bool)}|fw={int(fw_bool)}|hl={int(hl_bool)}|"
        f"cap={int(cap_bool)}|freeze={int(freeze_bool)}|shock={int(shock_bool)}"
    )


def format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_clock(dt):
    return dt.strftime("%H:%M:%S")


def sanitize_run_for_cache(run):
    return {
        "sector": run["sector"],
        "dates": list(pd.to_datetime(run["dates"])),
        "returns": {name: np.asarray(values, dtype=float) for name, values in run["returns"].items()},
        "exposure": {name: np.asarray(values, dtype=float) for name, values in run["exposure"].items()},
        "stats": dict(run.get("stats", {})),
        "strategy_stats": {name: dict(values) for name, values in run.get("strategy_stats", {}).items()},
        "trade_durations": list(run.get("trade_durations", [])),
        "trade_records": list(run.get("trade_records", [])),
        "strategy_trade_records": {
            name: list(records) for name, records in run.get("strategy_trade_records", {}).items()
        },
        "pair_stats": dict(run.get("pair_stats", {})),
        "selection_windows": list(run.get("selection_windows", [])),
        "strategy_selection_windows": list(run.get("strategy_selection_windows", [])),
        "active_pairs_timeline": np.asarray(run.get("active_pairs_timeline", []), dtype=int),
        "strategy_active_pairs_timeline": {
            name: np.asarray(values, dtype=int) for name, values in run.get("strategy_active_pairs_timeline", {}).items()
        },
        "strategy_pair_weight_timeline": {
            name: {pair: np.asarray(values, dtype=float) for pair, values in per_pair.items()}
            for name, per_pair in run.get("strategy_pair_weight_timeline", {}).items()
        },
        "cnf": dict(run.get("cnf", {})),
    }


def metric_value(metrics_dict, metric_name):
    if metric_name == "sharpe":
        return float(metrics_dict["sharpe"])
    if metric_name == "cagr":
        return float(metrics_dict["cagr"])
    if metric_name == "capital_final":
        return float(metrics_dict["final_cap"])
    if metric_name == "max_drawdown":
        return -float(metrics_dict["max_dd"])
    raise ValueError(f"Metrica desconocida: {metric_name}")


def summarize_trade_records(trade_records, start_index=0):
    filtered = [record for record in trade_records if int(record["start_index"]) >= start_index]
    durations = [int(record["duration"]) for record in filtered]
    return {
        "total_trades": int(len(filtered)),
        "avg_trade_duration": float(np.mean(durations)) if durations else 0.0,
        "median_trade_duration": float(np.median(durations)) if durations else 0.0,
        "max_trade_duration": int(np.max(durations)) if durations else 0,
        "min_trade_duration": int(np.min(durations)) if durations else 0,
    }


def summarize_window_choice_counts(window_choices):
    if not window_choices:
        return {
            "top_strategy": None,
            "top_combo_label": None,
            "strategy_counts": [],
            "combo_counts": [],
            "strategy_combo_counts": [],
        }

    df = pd.DataFrame(
        [
            {
                "strategy": choice["strategy"],
                "combo_label": choice["combo_label"],
            }
            for choice in window_choices
        ]
    )
    strategy_counts = (
        df.groupby("strategy", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "strategy"], ascending=[False, True])
        .reset_index(drop=True)
    )
    combo_counts = (
        df.groupby("combo_label", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "combo_label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    strategy_combo_counts = (
        df.groupby(["strategy", "combo_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "strategy", "combo_label"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return {
        "top_strategy": str(strategy_counts.iloc[0]["strategy"]) if not strategy_counts.empty else None,
        "top_combo_label": str(combo_counts.iloc[0]["combo_label"]) if not combo_counts.empty else None,
        "strategy_counts": strategy_counts.to_dict("records"),
        "combo_counts": combo_counts.to_dict("records"),
        "strategy_combo_counts": strategy_combo_counts.to_dict("records"),
    }


def choose_adaptive_sector_windows(sector_key, combo_runs, selection_metric):
    if not combo_runs:
        return None

    reference_combo, reference_run = next(iter(combo_runs.items()))
    dates = pd.to_datetime(reference_run["dates"])
    adaptive_returns = np.zeros(len(dates), dtype=float)
    adaptive_trade_records = []
    adaptive_windows = []

    reference_windows = reference_run.get("strategy_selection_windows", [])
    if not reference_windows:
        return None

    for window_idx in range(len(reference_windows)):
        best_candidate = None
        for combo, run in combo_runs.items():
            selection_windows = run.get("strategy_selection_windows", [])
            if window_idx >= len(selection_windows):
                continue
            window = selection_windows[window_idx]
            strategy_metrics = window.get("strategy_train_metrics", {})
            for strategy_name, metrics in strategy_metrics.items():
                score = metric_value(metrics, selection_metric)
                candidate = {
                    "combo": combo,
                    "combo_label": combo_to_label(combo),
                    "strategy": strategy_name,
                    "score": float(score),
                    "training_metrics": {
                        "cagr": float(metrics.get("cagr", 0.0)),
                        "sharpe": float(metrics.get("sharpe", 0.0)),
                        "max_dd": float(metrics.get("max_dd", 0.0)),
                        "final_cap": float(metrics.get("final_cap", 100.0)),
                    },
                    "out_idx_start": int(window.get("out_idx_start", 0)),
                    "out_idx_end": int(window.get("out_idx_end", 0)),
                    "training_start_date": window.get("training_start_date"),
                    "training_end_date": window.get("training_end_date"),
                    "operation_start_date": window.get("operation_start_date"),
                    "operation_end_date": window.get("operation_end_date"),
                    "window_index": int(window.get("window_index", window_idx + 1)),
                }
                if best_candidate is None or (
                    candidate["score"],
                    candidate["training_metrics"]["final_cap"],
                    candidate["training_metrics"]["sharpe"],
                ) > (
                    best_candidate["score"],
                    best_candidate["training_metrics"]["final_cap"],
                    best_candidate["training_metrics"]["sharpe"],
                ):
                    best_candidate = candidate

        if best_candidate is None:
            continue

        chosen_run = combo_runs[best_candidate["combo"]]
        out_idx_start = best_candidate["out_idx_start"]
        out_idx_end = best_candidate["out_idx_end"]
        strategy_name = best_candidate["strategy"]
        strategy_returns = np.asarray(chosen_run["returns"][strategy_name], dtype=float)
        adaptive_returns[out_idx_start:out_idx_end] = strategy_returns[out_idx_start:out_idx_end]

        strategy_trade_records = chosen_run.get("strategy_trade_records", {}).get(
            strategy_name,
            chosen_run.get("trade_records", []),
        )
        for record in strategy_trade_records:
            record_start = int(record.get("start_index", 0))
            if record_start < out_idx_start or record_start >= out_idx_end:
                continue
            copied = dict(record)
            copied["strategy"] = strategy_name
            copied["combo_label"] = best_candidate["combo_label"]
            adaptive_trade_records.append(copied)

        realized_metrics = calculate_metrics(strategy_returns[out_idx_start:out_idx_end])
        adaptive_windows.append(
            {
                **best_candidate,
                "realized_metrics": {
                    "cagr": float(realized_metrics["cagr"]),
                    "sharpe": float(realized_metrics["sharpe"]),
                    "max_dd": float(realized_metrics["max_dd"]),
                    "final_cap": float(realized_metrics["final_cap"]),
                },
            }
        )

    counts = summarize_window_choice_counts(adaptive_windows)
    trade_stats = summarize_trade_records(adaptive_trade_records, start_index=0)
    sector_metrics = calculate_metrics(adaptive_returns)
    return {
        "sector_key": sector_key,
        "run": reference_run,
        "returns": adaptive_returns,
        "trade_records": adaptive_trade_records,
        "window_choices": adaptive_windows,
        "top_strategy": counts["top_strategy"],
        "top_combo_label": counts["top_combo_label"],
        "strategy_counts": counts["strategy_counts"],
        "combo_counts": counts["combo_counts"],
        "strategy_combo_counts": counts["strategy_combo_counts"],
        "trade_stats": trade_stats,
        "sector_metrics": {
            "cagr": float(sector_metrics["cagr"]),
            "sharpe": float(sector_metrics["sharpe"]),
            "max_dd": float(sector_metrics["max_dd"]),
            "final_cap": float(sector_metrics["final_cap"]),
        },
        "pair_filter_stats": {
            "avg_pre_filtered_pairs": float(reference_run.get("stats", {}).get("avg_pre_filtered_pairs", 0.0)),
            "avg_strict_filtered_pairs": float(reference_run.get("stats", {}).get("avg_strict_filtered_pairs", 0.0)),
            "avg_selected_pairs": float(reference_run.get("stats", {}).get("avg_selected_pairs", 0.0)),
            "unique_selected_pairs": int(reference_run.get("stats", {}).get("unique_selected_pairs", 0)),
        },
    }


def choose_fixed_sector_choice(sector_key, combo_runs, selection_metric):
    if not combo_runs:
        return None

    best_training_choice = None
    best_oos_choice = None
    for combo, run in combo_runs.items():
        selection_windows = run.get("strategy_selection_windows", [])
        if not selection_windows:
            continue
        first_window = selection_windows[0]
        strategy_metrics = first_window.get("strategy_train_metrics", {})
        for strategy_name, train_metrics in strategy_metrics.items():
            if strategy_name not in run.get("returns", {}):
                continue

            strategy_returns = np.asarray(run["returns"][strategy_name], dtype=float)
            operation_metrics = calculate_metrics(strategy_returns)
            strategy_trade_records = run.get("strategy_trade_records", {}).get(
                strategy_name,
                run.get("trade_records", []),
            )
            candidate = {
                "sector_key": sector_key,
                "combo": combo,
                "combo_label": combo_to_label(combo),
                "strategy": strategy_name,
                "score": float(metric_value(train_metrics, selection_metric)),
                "training_metrics": {
                    "cagr": float(train_metrics.get("cagr", 0.0)),
                    "sharpe": float(train_metrics.get("sharpe", 0.0)),
                    "max_dd": float(train_metrics.get("max_dd", 0.0)),
                    "final_cap": float(train_metrics.get("final_cap", 100.0)),
                },
                "operation_metrics": {
                    "cagr": float(operation_metrics["cagr"]),
                    "sharpe": float(operation_metrics["sharpe"]),
                    "max_dd": float(operation_metrics["max_dd"]),
                    "final_cap": float(operation_metrics["final_cap"]),
                },
                "training_start_date": first_window.get("training_start_date"),
                "training_end_date": first_window.get("training_end_date"),
                "returns": strategy_returns,
                "trade_records": strategy_trade_records,
                "run": run,
                "pair_filter_stats": {
                    "avg_pre_filtered_pairs": float(run.get("stats", {}).get("avg_pre_filtered_pairs", 0.0)),
                    "avg_strict_filtered_pairs": float(run.get("stats", {}).get("avg_strict_filtered_pairs", 0.0)),
                    "avg_selected_pairs": float(run.get("stats", {}).get("avg_selected_pairs", 0.0)),
                    "unique_selected_pairs": int(run.get("stats", {}).get("unique_selected_pairs", 0)),
                },
            }
            if best_training_choice is None or (
                candidate["score"],
                candidate["training_metrics"]["final_cap"],
                candidate["training_metrics"]["sharpe"],
            ) > (
                best_training_choice["score"],
                best_training_choice["training_metrics"]["final_cap"],
                best_training_choice["training_metrics"]["sharpe"],
            ):
                best_training_choice = candidate

            if best_oos_choice is None or (
                candidate["operation_metrics"]["cagr"],
                candidate["operation_metrics"]["final_cap"],
                candidate["operation_metrics"]["sharpe"],
            ) > (
                best_oos_choice["operation_metrics"]["cagr"],
                best_oos_choice["operation_metrics"]["final_cap"],
                best_oos_choice["operation_metrics"]["sharpe"],
            ):
                best_oos_choice = candidate

    if best_training_choice is None:
        return None

    selected = dict(best_training_choice)
    selected["best_oos_choice"] = {
        "strategy": best_oos_choice["strategy"] if best_oos_choice else None,
        "combo_label": best_oos_choice["combo_label"] if best_oos_choice else None,
        "operation_metrics": dict(best_oos_choice["operation_metrics"]) if best_oos_choice else {},
    }
    selected["matches_best_oos"] = bool(
        best_oos_choice
        and selected["strategy"] == best_oos_choice["strategy"]
        and selected["combo_label"] == best_oos_choice["combo_label"]
    )
    return selected


def choose_best_by_sector(selection_sector_runs, operation_sector_runs, selection_metric):
    selected = {}

    for sector_key, combo_runs in selection_sector_runs.items():
        best_choice = None

        for combo, selection_run in combo_runs.items():
            if not combo[0]:
                continue
            operation_run = operation_sector_runs.get(sector_key, {}).get(combo)
            if operation_run is None:
                continue

            for strategy_name in ordered_names(selection_run["returns"].keys()):
                selection_returns = np.asarray(selection_run["returns"][strategy_name], dtype=float)
                operation_returns = np.asarray(operation_run["returns"][strategy_name], dtype=float)
                if len(selection_returns) < 5 or len(operation_returns) < 5:
                    continue

                selection_metrics = calculate_metrics(selection_returns)
                operation_metrics = calculate_metrics(operation_returns)
                score = metric_value(selection_metrics, selection_metric)

                strategy_trade_records = operation_run.get("strategy_trade_records", {}).get(
                    strategy_name,
                    operation_run.get("trade_records", []),
                )
                candidate = {
                    "sector": sector_key,
                    "combo": combo,
                    "combo_label": combo_to_label(combo),
                    "strategy": strategy_name,
                    "score": float(score),
                    "selection_metrics": {
                        "cagr": float(selection_metrics["cagr"]),
                        "sharpe": float(selection_metrics["sharpe"]),
                        "max_dd": float(selection_metrics["max_dd"]),
                        "final_cap": float(selection_metrics["final_cap"]),
                    },
                    "operation_metrics": {
                        "cagr": float(operation_metrics["cagr"]),
                        "sharpe": float(operation_metrics["sharpe"]),
                        "max_dd": float(operation_metrics["max_dd"]),
                        "final_cap": float(operation_metrics["final_cap"]),
                    },
                    "trade_stats_operation": summarize_trade_records(strategy_trade_records, start_index=0),
                    "pair_filter_stats_selection": {
                        "avg_pre_filtered_pairs": float(selection_run.get("stats", {}).get("avg_pre_filtered_pairs", 0.0)),
                        "avg_strict_filtered_pairs": float(selection_run.get("stats", {}).get("avg_strict_filtered_pairs", 0.0)),
                        "avg_selected_pairs": float(selection_run.get("stats", {}).get("avg_selected_pairs", 0.0)),
                        "unique_selected_pairs": int(selection_run.get("stats", {}).get("unique_selected_pairs", 0)),
                    },
                    "pair_filter_stats_operation": {
                        "avg_pre_filtered_pairs": float(operation_run.get("stats", {}).get("avg_pre_filtered_pairs", 0.0)),
                        "avg_strict_filtered_pairs": float(operation_run.get("stats", {}).get("avg_strict_filtered_pairs", 0.0)),
                        "avg_selected_pairs": float(operation_run.get("stats", {}).get("avg_selected_pairs", 0.0)),
                        "unique_selected_pairs": int(operation_run.get("stats", {}).get("unique_selected_pairs", 0)),
                    },
                }

                if best_choice is None or candidate["score"] > best_choice["score"]:
                    best_choice = candidate

        if best_choice is not None:
            selected[sector_key] = best_choice

    return selected


def build_selection_repetition_summary(selected_choices):
    if not selected_choices:
        return {
            "strategy_counts": [],
            "combo_counts": [],
            "strategy_combo_counts": [],
            "top_strategy": None,
            "top_combo": None,
            "top_strategy_combo": None,
        }

    rows = []
    for sector_key, choice in selected_choices.items():
        rows.append(
            {
                "sector": sector_key,
                "strategy": choice["strategy"],
                "combo_label": choice["combo_label"],
            }
        )

    df = pd.DataFrame(rows)
    strategy_counts = (
        df.groupby("strategy", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "strategy"], ascending=[False, True])
        .reset_index(drop=True)
    )
    combo_counts = (
        df.groupby("combo_label", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "combo_label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    strategy_combo_counts = (
        df.groupby(["strategy", "combo_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "strategy", "combo_label"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    return {
        "strategy_counts": strategy_counts.to_dict("records"),
        "combo_counts": combo_counts.to_dict("records"),
        "strategy_combo_counts": strategy_combo_counts.to_dict("records"),
        "top_strategy": strategy_counts.iloc[0].to_dict() if not strategy_counts.empty else None,
        "top_combo": combo_counts.iloc[0].to_dict() if not combo_counts.empty else None,
        "top_strategy_combo": strategy_combo_counts.iloc[0].to_dict() if not strategy_combo_counts.empty else None,
    }


def combine_sector_payloads(sector_payloads, benchmark_symbol="SPY"):
    if not sector_payloads:
        return None

    sector_keys = sorted(item["sector_key"] for item in sector_payloads)
    capital_fraction = 1.0 / len(sector_payloads)

    all_test_dates = set()
    contributions = {}
    total_trade_records = []
    total_trade_durations = []

    for item in sector_payloads:
        run = item["run"]
        run_dates = pd.to_datetime(run["dates"])
        all_test_dates.update(run_dates)
        records = list(item.get("trade_records", []))
        total_trade_records.extend(records)
        total_trade_durations.extend([int(record["duration"]) for record in records])

    index = pd.DatetimeIndex(sorted(all_test_dates))
    combined_capital = pd.Series(0.0, index=index)

    for item in sector_payloads:
        sector_key = item["sector_key"]
        run = item["run"]
        sector_returns = pd.Series(
            np.asarray(item["returns"], dtype=float),
            index=pd.to_datetime(run["dates"]),
        ).sort_index()
        sector_returns = sector_returns.reindex(index, fill_value=0.0)

        sector_capital_curve = 100.0 * capital_fraction * np.cumprod(1.0 + sector_returns.values)
        contributions[sector_key] = sector_capital_curve
        combined_capital = combined_capital.add(pd.Series(sector_capital_curve, index=index), fill_value=0.0)

    combined_returns = combined_capital.pct_change().fillna(0.0).values
    portfolio_metrics = calculate_metrics(combined_returns)
    portfolio_stats = {
        "total_trades": int(len(total_trade_records)),
        "avg_trade_duration": float(np.mean(total_trade_durations)) if total_trade_durations else 0.0,
        "median_trade_duration": float(np.median(total_trade_durations)) if total_trade_durations else 0.0,
        "max_trade_duration": int(np.max(total_trade_durations)) if total_trade_durations else 0,
        "min_trade_duration": int(np.min(total_trade_durations)) if total_trade_durations else 0,
    }

    benchmark = None
    if benchmark_symbol and yf is not None and len(index) > 2:
        try:
            start = (index.min() - timedelta(days=5)).strftime("%Y-%m-%d")
            end = (index.max() + timedelta(days=5)).strftime("%Y-%m-%d")
            data = yf.download(benchmark_symbol, start=start, end=end, auto_adjust=True, progress=False)
            if not data.empty:
                if "Close" in data.columns:
                    close = data["Close"]
                else:
                    close = data.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None)
                close = close.reindex(index, method="ffill").dropna()
                close = close.reindex(index, method="ffill")
                benchmark_curve = 100.0 * (close / close.iloc[0]).to_numpy(dtype=float)
                benchmark_returns = close.pct_change().fillna(0.0).values
                benchmark_metrics = calculate_metrics(benchmark_returns)
                benchmark = {
                    "label": benchmark_symbol,
                    "curve": benchmark_curve,
                    "metrics": {
                        "cagr": float(benchmark_metrics["cagr"]),
                        "sharpe": float(benchmark_metrics["sharpe"]),
                        "max_dd": float(benchmark_metrics["max_dd"]),
                        "final_cap": float(benchmark_metrics["final_cap"]),
                    },
                }
        except Exception as exc:
            benchmark = {"label": benchmark_symbol, "error": str(exc)}

    return {
        "dates": list(index.to_pydatetime()),
        "capital_curve": combined_capital.values,
        "returns": combined_returns,
        "metrics": {
            "cagr": float(portfolio_metrics["cagr"]),
            "sharpe": float(portfolio_metrics["sharpe"]),
            "max_dd": float(portfolio_metrics["max_dd"]),
            "final_cap": float(portfolio_metrics["final_cap"]),
        },
        "stats": portfolio_stats,
        "allocation_per_industry": float(100.0 * capital_fraction),
        "sector_contributions": {sector: list(curve) for sector, curve in contributions.items()},
        "benchmark": benchmark,
    }


def find_best_general_operation_choice(operation_sector_runs):
    if not operation_sector_runs:
        return None

    sector_keys = sorted(operation_sector_runs.keys())
    if not sector_keys:
        return None

    first_sector = sector_keys[0]
    combo_list = sorted(operation_sector_runs[first_sector].keys(), key=combo_to_label)
    strategy_names = ordered_names(operation_sector_runs[first_sector][combo_list[0]]["returns"].keys())
    rows = []

    for combo in combo_list:
        if any(combo not in operation_sector_runs[sector_key] for sector_key in sector_keys):
            continue

        for strategy_name in strategy_names:
            sector_payloads = []
            valid = True
            for sector_key in sector_keys:
                run = operation_sector_runs[sector_key][combo]
                if strategy_name not in run["returns"]:
                    valid = False
                    break
                strategy_trade_records = run.get("strategy_trade_records", {}).get(
                    strategy_name,
                    run.get("trade_records", []),
                )
                sector_payloads.append(
                    {
                        "sector_key": sector_key,
                        "run": run,
                        "returns": run["returns"][strategy_name],
                        "trade_records": strategy_trade_records,
                    }
                )

            if not valid:
                continue

            combined = combine_sector_payloads(sector_payloads, benchmark_symbol="")
            rows.append(
                {
                    "strategy": strategy_name,
                    "combo_label": combo_to_label(combo),
                    "cagr": float(combined["metrics"]["cagr"]),
                    "sharpe": float(combined["metrics"]["sharpe"]),
                    "max_dd": float(combined["metrics"]["max_dd"]),
                    "final_capital": float(combined["metrics"]["final_cap"]),
                    "total_trades": int(combined["stats"]["total_trades"]),
                }
            )

    if not rows:
        return None

    rows = sorted(
        rows,
        key=lambda row: (row["cagr"], row["final_capital"], row["sharpe"]),
        reverse=True,
    )
    return {
        "best": dict(rows[0]),
        "top_candidates": rows[:10],
    }


def build_general_portfolio(operation_sector_runs, selected_choices, benchmark_symbol="SPY", include_diagnostics=True):
    if not selected_choices:
        return None

    sector_keys = sorted(selected_choices.keys())
    sector_payloads = []
    for sector_key in sector_keys:
        choice = selected_choices[sector_key]
        run = operation_sector_runs[sector_key][choice["combo"]]
        strategy_trade_records = run.get("strategy_trade_records", {}).get(
            choice["strategy"],
            run.get("trade_records", []),
        )
        sector_payloads.append(
            {
                "sector_key": sector_key,
                "run": run,
                "returns": run["returns"][choice["strategy"]],
                "trade_records": strategy_trade_records,
            }
        )

    combined = combine_sector_payloads(sector_payloads, benchmark_symbol=benchmark_symbol)
    contribution_rows = []
    for sector_key in sector_keys:
        curve = np.asarray(combined["sector_contributions"][sector_key], dtype=float)
        choice = selected_choices[sector_key]
        contribution_rows.append(
            {
                "sector": sector_key,
                "final_capital": float(curve[-1]),
                "strategy": choice["strategy"],
                "combo_label": choice["combo_label"],
                "selection_score": float(choice["score"]),
                "selection_cagr": float(choice["selection_metrics"]["cagr"]),
                "selection_sharpe": float(choice["selection_metrics"]["sharpe"]),
                "selection_final_cap": float(choice["selection_metrics"]["final_cap"]),
                "operation_cagr": float(choice["operation_metrics"]["cagr"]),
                "operation_sharpe": float(choice["operation_metrics"]["sharpe"]),
                "operation_final_cap": float(choice["operation_metrics"]["final_cap"]),
                "avg_selected_pairs_selection": float(choice["pair_filter_stats_selection"]["avg_selected_pairs"]),
                "unique_selected_pairs_selection": int(choice["pair_filter_stats_selection"]["unique_selected_pairs"]),
                "avg_selected_pairs_operation": float(choice["pair_filter_stats_operation"]["avg_selected_pairs"]),
                "unique_selected_pairs_operation": int(choice["pair_filter_stats_operation"]["unique_selected_pairs"]),
            }
        )

    contribution_rows = sorted(contribution_rows, key=lambda row: row["final_capital"], reverse=True)
    highlights = {
        "top_contributor": contribution_rows[0]["sector"] if contribution_rows else None,
        "top_strategy": contribution_rows[0]["strategy"] if contribution_rows else None,
    }
    result = {
        **combined,
        "selected_choices": selected_choices,
        "contribution_table": contribution_rows,
        "highlights": highlights,
    }
    if include_diagnostics:
        result["selection_repetition"] = build_selection_repetition_summary(selected_choices)
        result["oos_best_common"] = find_best_general_operation_choice(operation_sector_runs)
    return result


def build_general_portfolio_fixed(operation_sector_runs, selection_metric, benchmark_symbol="SPY"):
    fixed_choices = {}
    for sector_key, combo_runs in operation_sector_runs.items():
        choice = choose_fixed_sector_choice(
            sector_key,
            combo_runs,
            selection_metric=selection_metric,
        )
        if choice is not None:
            fixed_choices[sector_key] = choice

    if not fixed_choices:
        return None

    sector_payloads = [
        {
            "sector_key": sector_key,
            "run": choice["run"],
            "returns": choice["returns"],
            "trade_records": choice["trade_records"],
        }
        for sector_key, choice in sorted(fixed_choices.items())
    ]
    combined = combine_sector_payloads(sector_payloads, benchmark_symbol=benchmark_symbol)

    contribution_rows = []
    selected_choices = {}
    for sector_key, choice in sorted(fixed_choices.items()):
        curve = np.asarray(combined["sector_contributions"][sector_key], dtype=float)
        best_oos = choice.get("best_oos_choice", {})
        best_oos_metrics = best_oos.get("operation_metrics", {})
        contribution_rows.append(
            {
                "sector": sector_key,
                "final_capital": float(curve[-1]),
                "strategy": choice["strategy"],
                "combo_label": choice["combo_label"],
                "selection_score": float(choice["score"]),
                "selection_cagr": float(choice["training_metrics"]["cagr"]),
                "selection_sharpe": float(choice["training_metrics"]["sharpe"]),
                "selection_final_cap": float(choice["training_metrics"]["final_cap"]),
                "operation_cagr": float(choice["operation_metrics"]["cagr"]),
                "operation_sharpe": float(choice["operation_metrics"]["sharpe"]),
                "operation_final_cap": float(choice["operation_metrics"]["final_cap"]),
                "best_oos_strategy": best_oos.get("strategy") or "-",
                "best_oos_combo_label": best_oos.get("combo_label") or "-",
                "best_oos_cagr": float(best_oos_metrics.get("cagr", 0.0)),
                "best_oos_final_cap": float(best_oos_metrics.get("final_cap", 100.0)),
                "matches_best_oos": bool(choice.get("matches_best_oos", False)),
                "training_start_date": choice.get("training_start_date"),
                "training_end_date": choice.get("training_end_date"),
                "avg_selected_pairs_operation": float(choice["pair_filter_stats"]["avg_selected_pairs"]),
                "unique_selected_pairs_operation": int(choice["pair_filter_stats"]["unique_selected_pairs"]),
            }
        )
        selected_choices[sector_key] = {
            "strategy": choice["strategy"],
            "combo_label": choice["combo_label"],
            "selection_score": float(choice["score"]),
            "training_metrics": dict(choice["training_metrics"]),
            "operation_metrics": dict(choice["operation_metrics"]),
            "best_oos_choice": best_oos,
            "matches_best_oos": bool(choice.get("matches_best_oos", False)),
            "training_start_date": choice.get("training_start_date"),
            "training_end_date": choice.get("training_end_date"),
            "selection_mode": "fixed_initial_training",
        }

    contribution_rows = sorted(contribution_rows, key=lambda row: row["final_capital"], reverse=True)
    return {
        **combined,
        "selected_choices": selected_choices,
        "contribution_table": contribution_rows,
        "highlights": {
            "top_contributor": contribution_rows[0]["sector"] if contribution_rows else None,
            "top_strategy": contribution_rows[0]["strategy"] if contribution_rows else None,
        },
        "selection_mode": "fixed_initial_training",
        "selection_metric": selection_metric,
    }


def build_general_portfolio_adaptive(operation_sector_runs, selection_metric, benchmark_symbol="SPY"):
    adaptive_sector_payloads = {}
    for sector_key, combo_runs in operation_sector_runs.items():
        payload = choose_adaptive_sector_windows(
            sector_key,
            combo_runs,
            selection_metric=selection_metric,
        )
        if payload is not None:
            adaptive_sector_payloads[sector_key] = payload

    if not adaptive_sector_payloads:
        return None

    sector_payloads = [
        {
            "sector_key": sector_key,
            "run": payload["run"],
            "returns": payload["returns"],
            "trade_records": payload["trade_records"],
        }
        for sector_key, payload in sorted(adaptive_sector_payloads.items())
    ]
    combined = combine_sector_payloads(sector_payloads, benchmark_symbol=benchmark_symbol)

    contribution_rows = []
    for sector_key, payload in sorted(adaptive_sector_payloads.items()):
        curve = np.asarray(combined["sector_contributions"][sector_key], dtype=float)
        contribution_rows.append(
            {
                "sector": sector_key,
                "final_capital": float(curve[-1]),
                "strategy": payload["top_strategy"] or "-",
                "combo_label": payload["top_combo_label"] or "-",
                "window_choices": list(payload["window_choices"]),
                "n_windows": int(len(payload["window_choices"])),
                "n_unique_strategies": int(len({choice["strategy"] for choice in payload["window_choices"]})),
                "n_unique_combos": int(len({choice["combo_label"] for choice in payload["window_choices"]})),
                "operation_cagr": float(payload["sector_metrics"]["cagr"]),
                "avg_selected_pairs_operation": float(payload["pair_filter_stats"]["avg_selected_pairs"]),
                "unique_selected_pairs_operation": int(payload["pair_filter_stats"]["unique_selected_pairs"]),
            }
        )

    contribution_rows = sorted(contribution_rows, key=lambda row: row["final_capital"], reverse=True)
    selected_choices = {
        sector_key: {
            "strategy": payload["top_strategy"] or "-",
            "combo_label": payload["top_combo_label"] or "-",
            "window_choices": list(payload["window_choices"]),
            "strategy_counts": list(payload["strategy_counts"]),
            "combo_counts": list(payload["combo_counts"]),
            "strategy_combo_counts": list(payload["strategy_combo_counts"]),
            "selection_mode": "adaptive_126d",
        }
        for sector_key, payload in adaptive_sector_payloads.items()
    }
    return {
        **combined,
        "selected_choices": selected_choices,
        "contribution_table": contribution_rows,
        "highlights": {
            "top_contributor": contribution_rows[0]["sector"] if contribution_rows else None,
            "top_strategy": contribution_rows[0]["strategy"] if contribution_rows else None,
        },
        "selection_mode": "adaptive_126d",
        "selection_metric": selection_metric,
    }


def read_ranked_sectors(ranking_file, top_n):
    if not os.path.isfile(ranking_file):
        return []
    df = pd.read_csv(ranking_file)
    key_col = None
    if "industry_key" in df.columns:
        key_col = "industry_key"
    elif "sector_key" in df.columns:
        key_col = "sector_key"
    if key_col is None:
        return []
    return df[key_col].dropna().astype(str).head(top_n).tolist()


def normalize_sector_list(sectors_raw, ranking_file, top_n):
    if sectors_raw:
        return [sector.strip().lower() for sector in sectors_raw.split(",") if sector.strip()]
    ranked = read_ranked_sectors(ranking_file, top_n)
    return [sector.lower() for sector in ranked]


def load_industry_index(index_file):
    if not os.path.isfile(index_file):
        raise FileNotFoundError(
            f"No existe {index_file}. Primero ejecuta build_industry_npz.py para generar la base por industria."
        )
    df = pd.read_csv(index_file)
    required = {"industry_key", "industry_name", "npz_file", "status", "start_date", "end_date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el indice de industrias: {sorted(missing)}")
    return df


def load_npz_dates(npz_file):
    with np.load(npz_file, allow_pickle=True) as dat:
        start = pd.to_datetime(str(dat["startdate"]))
        day = np.array(dat["day"])
    return pd.DatetimeIndex([start + pd.Timedelta(days=int(offset)) for offset in day])


def describe_period_coverage(sector_specs, period_start, period_end):
    period_start = pd.Timestamp(period_start)
    period_end = pd.Timestamp(period_end)
    rows = []

    for spec in sector_specs:
        dates = load_npz_dates(spec["npz_file"])
        dates = dates[(dates >= period_start) & (dates <= period_end)]
        if len(dates) == 0:
            rows.append(
                {
                    "sector_key": spec["sector_key"],
                    "count": 0,
                    "start": None,
                    "end": None,
                }
            )
        else:
            rows.append(
                {
                    "sector_key": spec["sector_key"],
                    "count": int(len(dates)),
                    "start": pd.Timestamp(dates[0]).date().isoformat(),
                    "end": pd.Timestamp(dates[-1]).date().isoformat(),
                }
            )

    return rows


def infer_training_count(npz_file, oos_start, training_years=4):
    dates = load_npz_dates(npz_file)
    if len(dates) == 0:
        raise ValueError(f"No hay fechas disponibles en {npz_file}.")

    requested_start = pd.Timestamp(oos_start)
    oos_dates = dates[dates >= requested_start]
    if len(oos_dates) == 0:
        raise ValueError(f"No existen fechas en {npz_file} a partir de {requested_start.date()}.")

    effective_oos_start = pd.Timestamp(oos_dates[0])
    training_start = effective_oos_start - pd.DateOffset(years=int(training_years))
    training_dates = dates[(dates >= training_start) & (dates < effective_oos_start)]
    actual_training_start = pd.Timestamp(training_dates[0]) if len(training_dates) else None
    actual_training_end = pd.Timestamp(training_dates[-1]) if len(training_dates) else None
    return int(len(training_dates)), effective_oos_start, actual_training_start, actual_training_end


def resolve_sector_specs(sectors, index_df):
    index_df = index_df.copy()
    index_df["industry_key_norm"] = index_df["industry_key"].astype(str).str.lower()
    index_df["industry_name_norm"] = index_df["industry_name"].astype(str).str.lower()
    ok_df = index_df[index_df["status"] == "ok"].copy()

    specs = []
    for sector in sectors:
        match = ok_df[
            (ok_df["industry_key_norm"] == sector.lower())
            | (ok_df["industry_name_norm"] == sector.lower())
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        npz_path = str(row["npz_file"])
        specs.append(
            {
                "sector_key": str(row["industry_key"]),
                "sector_name": str(row["industry_name"]),
                "pathdat": os.path.dirname(npz_path) or ".",
                "init_date": str(row["start_date"]),
                "end_date": str(row["end_date"]),
                "npz_file": npz_path,
            }
        )
    return specs


def result_file_for_sector(results_dir, sector_key):
    return os.path.join(results_dir, f"{sector_key}.pkl")


def sector_result_matches(payload, spec, combos):
    if not isinstance(payload, dict):
        return False
    if int(payload.get("schema_version", -1)) != INDUSTRY_RESULT_SCHEMA_VERSION:
        return False
    if payload.get("sector_key") != spec["sector_key"]:
        return False
    if payload.get("npz_file") != spec["npz_file"]:
        return False
    if payload.get("init_date") != spec["init_date"] or payload.get("end_date") != spec["end_date"]:
        return False

    expected_labels = [combo_to_label(combo) for combo in combos]
    cached_labels = payload.get("combo_labels", [])
    if cached_labels != expected_labels:
        return False

    operation_runs = payload.get("operation_runs")
    if not isinstance(operation_runs, dict):
        return False
    if set(operation_runs.keys()) != set(combos):
        return False
    return True


def load_sector_result(result_path, spec, combos):
    if not os.path.isfile(result_path):
        return None
    try:
        with open(result_path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not sector_result_matches(payload, spec, combos):
        return None
    return payload


def compute_sector_runs(spec, combos, oos_start, oos_end, freeze_stop_z, progress=None, phase_label="fase"):
    sector_runs = {}
    sector_start = time.perf_counter()
    inferred_training_count, effective_oos_start, actual_training_start, actual_training_end = infer_training_count(
        spec["npz_file"], oos_start, training_years=4
    )
    if inferred_training_count < 30:
        raise ValueError(
            f"{spec['sector_key']} no tiene suficientes observaciones para training antes de "
            f"{effective_oos_start.date()}: {inferred_training_count}"
        )
    print(
        f"Training inferido para {spec['sector_key']} [{phase_label}]: "
        f"{inferred_training_count} ruedas "
        f"({actual_training_start.date()} a {actual_training_end.date()})"
    )
    for combo_idx, combo in enumerate(combos, start=1):
        progress_text = ""
        if progress is not None and progress["total_steps"] > 0:
            if progress["completed_steps"] > 0:
                avg_step = progress["elapsed_total"] / progress["completed_steps"]
                remaining_steps = progress["total_steps"] - progress["completed_steps"]
                eta_text = format_duration(avg_step * remaining_steps)
            else:
                eta_text = "calculando..."
            progress_text = (
                f" | combo {combo_idx}/{len(combos)}"
                f" | progreso global {progress['completed_steps']}/{progress['total_steps']}"
                f" | ETA aprox {eta_text}"
            )
        print(f"Ejecutando {combo_to_label(combo)} [{phase_label}]{progress_text}")
        combo_start = time.perf_counter()
        cnf = build_conf_from_combo(combo, freeze_stop_z=freeze_stop_z)
        cnf.Ntraining = int(inferred_training_count)
        cnf.pathdat = spec["pathdat"]
        cnf.init_date = spec["init_date"]
        cnf.end_date = spec["end_date"]
        cnf.oos_start = oos_start
        cnf.oos_end = oos_end
        run = run_walk_forward(spec["sector_key"], cnf=cnf)
        sector_runs[combo] = sanitize_run_for_cache(run)
        combo_elapsed = time.perf_counter() - combo_start
        if progress is not None:
            progress["completed_steps"] += 1
            progress["elapsed_total"] += combo_elapsed
            if progress["completed_steps"] > 0:
                avg_step = progress["elapsed_total"] / progress["completed_steps"]
                remaining_steps = progress["total_steps"] - progress["completed_steps"]
                eta_text = format_duration(avg_step * remaining_steps)
            else:
                eta_text = "calculando..."
            print(
                f"Ultima combinacion completada {combo_idx}/{len(combos)} de {spec['sector_key']} [{phase_label}]"
                f" en {format_duration(combo_elapsed)} | ETA restante {eta_text}"
            )
    print(
        f"Sector {spec['sector_key']} completado para {phase_label} "
        f"en {format_duration(time.perf_counter() - sector_start)}"
    )
    return sector_runs


def save_sector_result(
    result_path,
    spec,
    combos,
    operation_runs,
    operation_start,
    operation_end,
    freeze_stop_z,
):
    operation_sample = next(iter(operation_runs.values()))
    operation_dates = pd.DatetimeIndex(pd.to_datetime(operation_sample["dates"]))
    payload = {
        "schema_version": INDUSTRY_RESULT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sector_key": spec["sector_key"],
        "sector_name": spec["sector_name"],
        "pathdat": spec["pathdat"],
        "init_date": spec["init_date"],
        "end_date": spec["end_date"],
        "npz_file": spec["npz_file"],
        "combo_labels": [combo_to_label(combo) for combo in combos],
        "operation_start": pd.Timestamp(operation_start).date().isoformat(),
        "operation_end": pd.Timestamp(operation_end).date().isoformat(),
        "effective_operation_start": pd.Timestamp(operation_dates[0]).date().isoformat() if len(operation_dates) else None,
        "effective_operation_end": pd.Timestamp(operation_dates[-1]).date().isoformat() if len(operation_dates) else None,
        "effective_operation_count": int(len(operation_dates)),
        "operation_runs": operation_runs,
        "freeze_stop_z": float(freeze_stop_z),
    }
    with open(result_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def ensure_sector_result(
    spec,
    combos,
    results_dir,
    operation_start,
    operation_end,
    freeze_stop_z,
    force_recompute=False,
    progress=None,
):
    os.makedirs(results_dir, exist_ok=True)
    result_path = result_file_for_sector(results_dir, spec["sector_key"])

    if not force_recompute:
        cached_payload = load_sector_result(result_path, spec, combos)
        if cached_payload is not None:
            if (
                cached_payload.get("operation_start") == pd.Timestamp(operation_start).date().isoformat()
                and cached_payload.get("operation_end") == pd.Timestamp(operation_end).date().isoformat()
                and float(cached_payload.get("freeze_stop_z", np.nan)) == float(freeze_stop_z)
            ):
                print(f"Usando cache existente: {result_path}")
                return cached_payload, False
            print(f"Cache desactualizado por cambio de periodos: {result_path}")

    operation_runs = compute_sector_runs(
        spec,
        combos,
        operation_start,
        operation_end,
        freeze_stop_z,
        progress=progress,
        phase_label="operacion",
    )
    payload = save_sector_result(
        result_path,
        spec,
        combos,
        operation_runs,
        operation_start,
        operation_end,
        freeze_stop_z,
    )
    print(f"Guardado cache por industria: {result_path}")
    return payload, True


def main():
    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()
    args = parse_args()
    sectors = normalize_sector_list(args.sectors, args.ranking_file, args.top_industries)
    index_df = load_industry_index(args.index_file)
    sector_specs = resolve_sector_specs(sectors, index_df)

    if not sector_specs:
        raise SystemExit(
            "No hay industrias para preparar. Ejecuta primero rank_industries.py o pasa --sectors manualmente."
        )

    combos = build_combo_grid()
    operation_sector_runs = {}
    sector_result_files = {}
    computed_count = 0
    reused_count = 0
    cached_payloads = {}
    operation_coverage = describe_period_coverage(sector_specs, args.operation_start, args.operation_end)

    if not args.force_recompute:
        for spec in sector_specs:
            result_path = result_file_for_sector(args.results_dir, spec["sector_key"])
            payload = load_sector_result(result_path, spec, combos)
            if (
                payload is not None
                and payload.get("operation_start") == pd.Timestamp(args.operation_start).date().isoformat()
                and payload.get("operation_end") == pd.Timestamp(args.operation_end).date().isoformat()
                and float(payload.get("freeze_stop_z", np.nan)) == float(args.freeze_stop_z)
            ):
                cached_payloads[spec["sector_key"]] = payload

    sectors_to_compute = [spec for spec in sector_specs if spec["sector_key"] not in cached_payloads]
    progress = {
        "total_steps": len(sectors_to_compute) * len(combos),
        "completed_steps": 0,
        "elapsed_total": 0.0,
    }

    print("=== Preparando cache para app.py ===")
    print(f"Inicio de ejecucion: {format_clock(started_at)}")
    print("Industrias:", ", ".join(spec["sector_key"] for spec in sector_specs))
    print(f"Combinaciones por industria: {len(combos)}")
    print(f"Cache por industria: {args.results_dir}")
    print(f"Umbral stop mu/sigma fijos: {args.freeze_stop_z:.2f}")
    print("Periodo walk-forward:", f"{pd.Timestamp(args.operation_start).date()} a {pd.Timestamp(args.operation_end).date()}")
    for row in operation_coverage:
        print(
            f"  - {row['sector_key']}: {row['count']} ruedas"
            + (
                f" ({row['start']} a {row['end']})"
                if row["count"] > 0
                else " (sin datos en el rango)"
            )
        )
    print(f"Industrias a recalcular: {len(sectors_to_compute)}")
    print(f"Industrias reutilizadas: {len(cached_payloads)}")

    for spec in sector_specs:
        sector_key = spec["sector_key"]
        print(f"\n--- Sector {sector_key.upper()} ---")
        if sector_key in cached_payloads:
            payload = cached_payloads[sector_key]
            was_computed = False
            print(f"Usando cache existente: {result_file_for_sector(args.results_dir, sector_key)}")
        else:
            payload, was_computed = ensure_sector_result(
                spec,
                combos,
                results_dir=args.results_dir,
                operation_start=args.operation_start,
                operation_end=args.operation_end,
                freeze_stop_z=args.freeze_stop_z,
                force_recompute=args.force_recompute,
                progress=progress,
            )
        operation_sector_runs[sector_key] = payload["operation_runs"]
        sector_result_files[sector_key] = result_file_for_sector(args.results_dir, sector_key)
        computed_count += int(was_computed)
        reused_count += int(not was_computed)

    general_fixed_summary = build_general_portfolio_fixed(
        operation_sector_runs,
        selection_metric=args.selection_metric,
        benchmark_symbol=args.benchmark,
    )

    general_summary = build_general_portfolio_adaptive(
        operation_sector_runs,
        selection_metric=args.selection_metric,
        benchmark_symbol=args.benchmark,
    )

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection_metric": args.selection_metric,
            "sectors": [spec["sector_key"] for spec in sector_specs],
            "sector_names": {spec["sector_key"]: spec["sector_name"] for spec in sector_specs},
            "sector_result_files": sector_result_files,
            "industry_results_dir": args.results_dir,
            "selection_mode": "adaptive_126d",
            "rebalance_window_days": 126,
            "operation_start": pd.Timestamp(args.operation_start).date().isoformat(),
            "operation_end": pd.Timestamp(args.operation_end).date().isoformat(),
            "operation_coverage": operation_coverage,
            "freeze_stop_z": float(args.freeze_stop_z),
            "combos": [{"combo": combo, "label": combo_to_label(combo)} for combo in combos],
        },
        "sector_runs": operation_sector_runs,
        "general_fixed": general_fixed_summary,
        "general": general_summary,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(payload, handle)

    print(f"\nCache guardado en: {args.output}")
    print(f"Industrias recalculadas: {computed_count}")
    print(f"Industrias reutilizadas desde cache: {reused_count}")
    if general_summary is not None:
        print("\n=== Seleccion adaptativa por industria ===")
        for sector_key, choice in general_summary.get("selected_choices", {}).items():
            print(
                f"{sector_key:15s} -> estrategia dominante {choice['strategy']:10s} | "
                f"{choice['combo_label']}"
            )
        print(
            f"\nCapital final combinado: {general_summary['metrics']['final_cap']:.2f} | "
            f"CAGR: {general_summary['metrics']['cagr'] * 100:.2f}% | "
            f"Sharpe: {general_summary['metrics']['sharpe']:.2f}"
        )
    if general_fixed_summary is not None:
        print("\n=== Seleccion fija por industria ===")
        for sector_key, choice in general_fixed_summary.get("selected_choices", {}).items():
            best_oos = choice.get("best_oos_choice", {})
            print(
                f"{sector_key:15s} -> estrategia fija {choice['strategy']:10s} | "
                f"{choice['combo_label']} | mejor OOS "
                f"{best_oos.get('strategy', '-')} | {best_oos.get('combo_label', '-')}"
            )
        print(
            f"\nCapital final combinado fijo: {general_fixed_summary['metrics']['final_cap']:.2f} | "
            f"CAGR: {general_fixed_summary['metrics']['cagr'] * 100:.2f}% | "
            f"Sharpe: {general_fixed_summary['metrics']['sharpe']:.2f}"
        )

    finished_at = datetime.now().astimezone()
    elapsed = time.perf_counter() - started_perf
    print("\n=== Tiempo total de ejecucion ===")
    print(f"Inicio: {format_clock(started_at)}")
    print(f"Fin:    {format_clock(finished_at)}")
    print(f"Duracion: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
