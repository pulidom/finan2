import numpy as np
import pandas as pd

EPSILON = 1e-8

def _normalize(w_raw):
    """Normaliza los pesos para que sumen 1 (100% del portafolio) por cada día."""
    s = w_raw.sum(axis=0)
    s_safe = np.where(s == 0, 1, s)  # Evita división por cero
    return w_raw / s_safe

def _scale_to_target_capital(w_raw, target_capital=1.0):
    """Normaliza y luego escala la exposición total diaria al capital objetivo."""
    return _normalize(w_raw) * target_capital


def signed_turnover_event_scores(price_matrix, volume_matrix, window=126, z_threshold=2.5):
    """
    Construye un score firmado de shock externo a partir de:

        score_t = sign(ret_t) * max(zscore(log(1 + P_t * Vol_t)) - z_threshold, 0)

    El z-score se calcula usando media y desvío rolling con shift(1), de modo que
    en el día t solo se usan datos hasta t-1 para la estandarización.
    """
    price_matrix = np.asarray(price_matrix, dtype=float)
    volume_matrix = np.asarray(volume_matrix, dtype=float)

    if price_matrix.shape != volume_matrix.shape:
        raise ValueError("price_matrix y volume_matrix deben tener la misma forma.")

    if price_matrix.ndim != 2:
        raise ValueError("price_matrix debe ser una matriz 2D de activos x tiempo.")

    window = max(int(window), 2)

    turnover = np.log1p(np.maximum(price_matrix * volume_matrix, 0.0))
    turnover_df = pd.DataFrame(turnover.T)
    price_df = pd.DataFrame(price_matrix.T)

    rolling_mean = turnover_df.rolling(window=window, min_periods=window).mean().shift(1)
    rolling_std = turnover_df.rolling(window=window, min_periods=window).std().shift(1)
    rolling_std = rolling_std.mask(~np.isfinite(rolling_std) | (rolling_std <= EPSILON), np.nan)

    zturn = (turnover_df - rolling_mean) / rolling_std
    zturn = zturn.where(np.isfinite(zturn), 0.0)
    zturn_pos = (zturn - float(z_threshold)).clip(lower=0.0)

    returns = price_df.pct_change().fillna(0.0)
    signed_scores = np.sign(returns) * zturn_pos

    return signed_scores.to_numpy(dtype=float).T


def apply_event_shock_overlay(base_weights, adverse_event_scores, penalty_lambda=0.5):
    """
    Penaliza pesos cuando el shock externo va en contra de la posición del par,
    pero preserva la exposición total diaria de la estrategia base.
    """
    base_weights = np.asarray(base_weights, dtype=float)
    adverse_event_scores = np.asarray(adverse_event_scores, dtype=float)

    if base_weights.shape != adverse_event_scores.shape:
        raise ValueError("base_weights y adverse_event_scores deben tener la misma forma.")

    penalty_lambda = max(float(penalty_lambda), 0.0)
    penalties = np.exp(-penalty_lambda * np.maximum(adverse_event_scores, 0.0))
    adjusted = base_weights * penalties

    base_total = np.sum(base_weights, axis=0)
    adjusted_total = np.sum(adjusted, axis=0)
    scale = np.divide(
        base_total,
        adjusted_total,
        out=np.zeros_like(base_total, dtype=float),
        where=adjusted_total > EPSILON,
    )

    return adjusted * scale

def _sanitize_half_life(hl_val):
    if np.isnan(hl_val) or np.isinf(hl_val) or hl_val < 1:
        return 1
    return int(round(hl_val))

def _apply_priority_cap(w_matrix, locked_mask, max_total=1.0):
    """
    Respeta primero los pesos bloqueados por half-life y solo reduce los libres.
    Si los bloqueados por sí solos exceden el capital máximo, se escala como último recurso.
    """
    if max_total is None:
        return w_matrix

    w_capped = w_matrix.copy()
    n_days = w_capped.shape[1]

    for t in range(n_days):
        total_t = w_capped[:, t].sum()
        if total_t <= max_total + EPSILON:
            continue

        locked_t = locked_mask[:, t]
        free_t = ~locked_t

        locked_sum = w_capped[locked_t, t].sum()
        free_sum = w_capped[free_t, t].sum()

        if locked_sum >= max_total and locked_sum > EPSILON:
            w_capped[locked_t, t] *= max_total / locked_sum
            w_capped[free_t, t] = 0.0
            continue

        remaining = max_total - locked_sum
        if free_sum > EPSILON and remaining < free_sum:
            w_capped[free_t, t] *= max(remaining, 0.0) / free_sum

    return w_capped

def equal_weight(positions, target_capital=1.0):
    w_raw = np.abs(positions)
    return _scale_to_target_capital(w_raw, target_capital)

def risk_parity_weight(positions, spread_vols, target_capital=1.0):
    w_raw = np.abs(positions) / (spread_vols[:, None] + EPSILON)
    return _scale_to_target_capital(w_raw, target_capital)

def inverse_variance_weight(positions, spread_vols, target_capital=1.0):
    w_raw = np.abs(positions) / ((spread_vols[:, None] ** 2) + EPSILON)
    return _scale_to_target_capital(w_raw, target_capital)

def zscore_pure_weight(positions, z_matrix, target_capital=1.0):
    w_raw = np.abs(positions * z_matrix)
    return _scale_to_target_capital(w_raw, target_capital)

def zscore_squashed_weight(positions, z_matrix, spread_vols, target_capital=1.0):
    z_squashed = np.tanh(np.abs(z_matrix) / 2.0)
    w_raw = (np.abs(positions) * z_squashed) / (spread_vols[:, None] + EPSILON)
    return _scale_to_target_capital(w_raw, target_capital)

def crossings_weight(positions, z_matrix, spread_vols, crossings_rate, target_capital=1.0):
    w_raw = (np.abs(positions * z_matrix) * crossings_rate[:, None]) / (spread_vols[:, None] + EPSILON)
    return _scale_to_target_capital(w_raw, target_capital)

def ggr_weight(positions, z_matrix, spread_vols, target_capital=1.0, clip=0.3, vol_floor=0.01):
    """
    Implementación tipo Gatev/Goetzmann/Rouwenhorst:
    la dirección la aporta la posición y el z-score solo aporta magnitud.
    """
    n_dinamico = np.maximum((np.abs(positions) > 0).sum(axis=0), 1)
    sigma = np.maximum(spread_vols[:, None], vol_floor)
    w_raw = (np.abs(z_matrix) * np.abs(positions)) / (2.0 * sigma * n_dinamico)

    if clip is not None:
        w_raw = np.clip(w_raw, 0.0, clip)

    return _scale_to_target_capital(w_raw, target_capital)

def kelly_dynamic_weight(positions, z_matrix, spread_vols, kelly_fraction=1.0, clip=0.3, vol_floor=0.01):
    # n dinámico: cantidad de activos con posición abierta en cada día específico
    n_dinamico = (np.abs(positions) > 0).sum(axis=0)
    n_dinamico_safe = np.where(n_dinamico == 0, 1, n_dinamico)
    
    # Fórmula: w = |-z / (2 * sigma * n)|
    sigma = np.maximum(spread_vols[:, None], vol_floor)
    w_raw = (np.abs(z_matrix) / (2 * sigma * n_dinamico_safe)) * np.abs(positions)
    
    # Aplicar Fractional Kelly
    w_raw = w_raw * kelly_fraction

    if clip is not None:
        w_raw = np.clip(w_raw, 0.0, clip)
    
    # A diferencia de otras estrategias, Kelly dicta un tamaño de apuesta absoluto.
    # Evitamos forzar la suma a 1 siempre (si kelly pide invertir 30%, invertimos 30%).
    # Sin embargo, si la suma supera el 100%, normalizamos para no apalancar de más.
    s = w_raw.sum(axis=0)
    factor = np.where(s > 1.0, s, 1.0)
    
    return w_raw / factor

def kelly_ou_weight(
    positions,
    z_matrix,
    returns_matrix,
    spread_vols,
    lookback=20,
    kelly_fraction=0.5,
    clip=0.3,
    vol_floor=0.01,
    min_signal=1.0,
    target_capital=1.0,
):
    """
    Kelly con mu/sigma estimados desde una dinámica OU/AR(1) del z-score,
    inspirado en el material comparativo del trabajo externo.
    """
    n_pairs, n_days = positions.shape
    w_raw = np.zeros((n_pairs, n_days))

    for t in range(n_days):
        active = np.abs(positions[:, t]) > 0
        if not np.any(active):
            continue

        idx = np.where(active)[0]
        z_t = np.abs(z_matrix[idx, t])

        if t >= lookback:
            z_hist = z_matrix[idx, t - lookback:t]
            z_lag = z_hist[:, :-1]
            z_lead = z_hist[:, 1:]
            cov = np.mean(z_lead * z_lag, axis=1) - np.mean(z_lead, axis=1) * np.mean(z_lag, axis=1)
            var = np.var(z_lag, axis=1) + EPSILON
            b = np.clip(cov / var, 0.001, 0.9999)
            theta = -np.log(b)
            vol_ret = np.std(returns_matrix[idx, t - lookback:t], axis=1)
        else:
            theta = np.full(len(idx), 0.05)
            if t <= 0:
                vol_ret = np.maximum(spread_vols[idx], vol_floor)
            else:
                vol_ret = np.std(returns_matrix[idx, :t], axis=1)

        mu = theta * z_t
        mu[z_t < min_signal] = 0.0

        # El riesgo principal se estima desde el retorno del par; el spread_vol se usa como piso adicional.
        vol = np.maximum(vol_ret, np.maximum(spread_vols[idx], vol_floor))
        w_kelly = kelly_fraction * mu / (vol ** 2)

        if clip is not None:
            w_kelly = np.clip(w_kelly, 0.0, clip)

        w_raw[idx, t] = w_kelly

    return _apply_priority_cap(w_raw, np.zeros_like(w_raw, dtype=bool), max_total=target_capital)

def zscore_inverse_weight(positions, z_matrix, epsilon=0.5, target_capital=1.0):
    """
    Estrategia Inversa: Asigna más capital a medida que el z-score 
    se acerca a 0, y reduce la exposición en divergencias extremas.
    """
    # Se suma epsilon para evitar la división por cero y suavizar la curva
    w_raw = (1.0 / (np.abs(z_matrix) + epsilon)) * np.abs(positions)
    
    return _scale_to_target_capital(w_raw, target_capital)

def past_performance_weight(positions, past_returns, min_weight=0.1, target_capital=1.0):
    """
    Asigna pesos basados en el retorno previo del par.
    - past_returns: array con los rendimientos históricos porcentuales de los pares 
      (ej. en el período de training).
    - Los retornos negativos se castigan (se asigna min_weight constante general o disminuye gradualmente),
      pero no se pone en 0 para permitir reversión a la media.
    """
    # Exponencial o Softmax escalado simple:
    # Si past_return > 0 -> w ~ past_return
    # Si past_return <= 0 -> w = min_weight
    
    # Normalizamos los returns pasados para tener una base 1.0 = Max Return
    if past_returns.max() > 0:
        norm_ret = past_returns / past_returns.max()
    else:
        norm_ret = np.ones_like(past_returns)
        
    base_w = np.where(norm_ret > 0, norm_ret, min_weight)
    
    # Ponderamos la matriz de posiciones con este vector de base_w
    w_raw = np.abs(positions) * base_w[:, None]
    
    return _scale_to_target_capital(w_raw, target_capital)




def apply_hold_period(w_matrix, positions, half_lives, max_total=1.0):
    """
    Mantiene como piso el peso tomado al entrar hasta completar el half-life del par.
    Si la posición se cierra por z-score, se cierra inmediatamente.
    Solo se permiten reducciones antes del half-life si es estrictamente necesario
    para no exceder el capital total disponible.
    """
    w_held = np.zeros_like(w_matrix)
    locked_mask = np.zeros_like(w_matrix, dtype=bool)
    n_pairs, n_days = w_matrix.shape
    
    for i in range(n_pairs):
        hl = _sanitize_half_life(half_lives[i])
        peso_entrada = 0.0
        inicio_trade = None
        pos_prev = 0.0
        
        for t in range(n_days):
            pos_t = positions[i, t]
            nueva_entrada = pos_t != 0 and (pos_prev == 0 or np.sign(pos_t) != np.sign(pos_prev))

            if pos_t != 0:
                if nueva_entrada:
                    inicio_trade = t
                    peso_entrada = w_matrix[i, t]

                dias_abierto = (t - inicio_trade + 1) if inicio_trade is not None else 1

                if dias_abierto <= hl:
                    w_held[i, t] = max(w_matrix[i, t], peso_entrada)
                    locked_mask[i, t] = True
                else:
                    w_held[i, t] = w_matrix[i, t]
            else:
                inicio_trade = None
                peso_entrada = 0.0
                w_held[i, t] = 0.0

            pos_prev = pos_t

    return _apply_priority_cap(w_held, locked_mask, max_total=max_total)

import numpy as np

def calcular_tope_por_promedio_señales(positions_train):
    """
    Calcula el promedio de posiciones simultáneas (n) en días activos 
    y devuelve el tope máximo permitido por par (1/n).
    """
    # 1. Sumamos la cantidad absoluta de posiciones abiertas por cada día
    señales_diarias = np.sum(np.abs(positions_train), axis=0)
    
    # 2. Filtramos solo los días donde el algoritmo estuvo dentro del mercado
    dias_activos = señales_diarias[señales_diarias > 0]
    
    if len(dias_activos) == 0:
        return 1.0, 1.0 # Seguridad: si nunca operó, devuelve 100% y n=1
        
    # 3. Calculamos el promedio de posiciones simultáneas (n)
    n_promedio = np.mean(dias_activos)
    
    # 4. Establecemos el tope como 1 / n
    # Usamos max(1, n) por seguridad, para nunca asignar más del 100% a un par
    tope_maximo = 1.0 / max(1.0, n_promedio)
    
    return tope_maximo, n_promedio
