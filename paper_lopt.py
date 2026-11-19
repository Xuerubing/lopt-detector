from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import log_ndtr, logsumexp


@dataclass(frozen=True)
class Config:
    layer_count: int = 36
    history_steps: int = 50
    shift_count: int = 8
    min_std: float = 1e-12
    d0_quantile: float = 0.90
    gate_quantile: float = 0.90
    lambda_weight: float = 1.0
    beta_weight: float = 1.0
    alpha_total: float = 0.003
    stream_count: int = 3


def calibrate(history, config=Config()):
    values = np.asarray(history, dtype=np.float64)
    if values.shape != (config.history_steps, config.layer_count):
        raise ValueError("Wrong history shape")
    if not np.isfinite(values).all() or config.history_steps <= config.shift_count:
        raise ValueError("History must be finite and longer than shifts")
    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0, ddof=1), config.min_std)
    normalized = (values - center) / scale
    differences = np.concatenate([
        (normalized - np.roll(normalized, shift=shift, axis=0)).ravel()
        for shift in range(1, config.shift_count + 1)
    ])
    row_means = normalized.mean(axis=1)
    calibration = {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "sigma": max(float(differences.std(ddof=1)), config.min_std),
        "d0": max(0.0, float(np.quantile(differences, config.d0_quantile, method="linear"))),
        "gate_boundary": max(0.0, float(np.quantile(row_means - row_means.mean(), config.gate_quantile, method="linear"))),
        "difference_count": int(differences.size),
    }
    if not all(math.isfinite(calibration[key]) for key in ("sigma", "d0", "gate_boundary")):
        raise ValueError("Nonfinite calibration")
    return calibration


def path_orders(layer_count):
    if layer_count < 1:
        raise ValueError("No layers")
    identity = list(range(layer_count))
    midpoint = layer_count // 2
    return [identity, identity[::-1], identity[midpoint:] + identity[:midpoint][::-1]]


def path_trace(differences, order, sigma, d0, lambda_weight=1.0, beta_weight=1.0):
    values = np.asarray(differences, dtype=np.float64)
    layer_count = len(values)
    if not layer_count or sorted(order) != list(range(layer_count)) or not np.isfinite(values).all():
        raise ValueError("Invalid path input")
    if sigma <= 0 or d0 < 0 or lambda_weight <= 0 or beta_weight <= 0:
        raise ValueError("Invalid path constants")
    selected_values = []
    standardized_values = []
    trace = []
    score = 0.0
    for position, layer in enumerate(order, start=1):
        direction = 1 if score >= 0 else -1
        difference = float(values[layer])
        selected = direction * difference
        selected_center = direction * d0
        selected_values.append(selected)
        standardized_values.append((selected - selected_center) / sigma)
        lln = lambda_weight * math.fsum(selected_values) / layer_count
        clt = beta_weight * math.fsum(standardized_values) / math.sqrt(layer_count)
        score = lln + clt
        if not math.isfinite(score):
            raise ValueError("Nonfinite path statistic")
        trace.append({
            "position": position, "layer": layer, "direction": direction,
            "difference": difference, "selected": selected, "selected_center": selected_center,
            "lln": lln, "clt": clt, "score": score,
        })
    return score, trace


def spike_tail(score, d0, sigma, beta_weight=1.0):
    if not all(math.isfinite(value) for value in (score, d0, sigma, beta_weight)):
        raise ValueError("Nonfinite tail arguments")
    if d0 < 0 or sigma <= 0 or beta_weight <= 0:
        raise ValueError("Invalid tail arguments")
    if score == 0:
        return {"p_value": 1.0, "log_p_value": 0.0}
    scale = beta_weight * math.hypot(1.0, d0 / sigma)
    radius = abs(score) / scale
    log_probability = min(0.0, float(logsumexp([
        log_ndtr(-d0 - radius),
        -2.0 * d0 * radius + log_ndtr(d0 - radius),
    ])))
    if not math.isfinite(log_probability):
        raise ValueError("Tail outside supported floating range")
    return {"p_value": math.exp(log_probability), "log_p_value": log_probability}


def cauchy_combine(probabilities):
    if not probabilities or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Invalid p-values")
    clipped = [min(1.0 - 1e-15, max(1e-300, value)) for value in probabilities]
    contributions = []
    for probability in clipped:
        if probability < 0.5:
            contributions.append(1.0 / math.tan(math.pi * probability))
        elif probability > 0.5:
            contributions.append(-1.0 / math.tan(math.pi * (1.0 - probability)))
        else:
            contributions.append(0.0)
    combined = math.fsum(contributions) / len(contributions)
    if combined > 0:
        probability = math.atan(1.0 / combined) / math.pi
    elif combined < 0:
        probability = 1.0 + math.atan(1.0 / combined) / math.pi
    else:
        probability = 0.5
    return {"p_value": probability, "cauchy_statistic": combined, "endpoint_clipped": clipped != list(probabilities)}


def test_stream(current, previous, calibration, config=Config()):
    current = np.asarray(current, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if current.shape != (config.layer_count,) or previous.shape != current.shape:
        raise ValueError("Wrong current/previous shape")
    nonfinite = np.flatnonzero(~np.isfinite(current) | ~np.isfinite(previous)).tolist()
    if nonfinite:
        return {"status": "NONFINITE", "nonfinite_layers": nonfinite, "alarm": None, "p_value": None, "paths": []}
    center = np.asarray(calibration["center"])
    scale = np.asarray(calibration["scale"])
    if center.shape != current.shape or scale.shape != current.shape or not np.isfinite(center).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid layer calibration")
    current_domain = (current - center) / scale
    previous_domain = (previous - center) / scale
    differences = current_domain - previous_domain
    current_mean = math.fsum(current_domain) / config.layer_count
    previous_mean = math.fsum(previous_domain) / config.layer_count
    mean_shift = current_mean - previous_mean
    gate_boundary = calibration["gate_boundary"]
    result = {
        "current_mean": current_mean, "previous_mean": previous_mean,
        "mean_shift": mean_shift, "gate_boundary": gate_boundary,
        "gate_margin": mean_shift - gate_boundary,
        "sigma": calibration["sigma"], "d0": calibration["d0"],
        "alpha_per_stream": config.alpha_total / config.stream_count,
        "normalized_differences": differences.tolist(),
        "nonfinite_layers": [], "paths": [], "p_value": None,
    }
    if mean_shift <= gate_boundary:
        return {**result, "status": "GATE_REJECT", "gate_passed": False, "alarm": False}
    for order in path_orders(config.layer_count):
        score, trace = path_trace(differences, order, calibration["sigma"], calibration["d0"], config.lambda_weight, config.beta_weight)
        tail = spike_tail(score, calibration["d0"], calibration["sigma"], config.beta_weight)
        result["paths"].append({"order": order, "score": score, "trace": trace, **tail})
    combined = cauchy_combine([path["p_value"] for path in result["paths"]])
    return {
        **result, **combined, "status": "PATH_EVALUATED", "gate_passed": True,
        "alarm": combined["p_value"] < result["alpha_per_stream"],
    }


def joint_decision(stream_results):
    decisions = [stream["alarm"] for stream in stream_results.values()]
    if not decisions:
        raise ValueError("No monitor decisions")
    if any(value is True for value in decisions):
        return True
    if any(value is None for value in decisions):
        return None
    return False
