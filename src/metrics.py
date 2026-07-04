"""Forecast error metrics.

MAE is the primary metric; the rest give scale-free and relative views. MASE is
scaled by the seasonal naive (lag-24) in-sample error so a value below 1 means
the model beats the everyday "same hour yesterday" heuristic.
"""

from __future__ import annotations

import numpy as np

SEASONAL_PERIOD = 24


def _to_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="float64")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, the primary metric."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error; penalises large misses more than MAE."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%), skipping zero actuals to stay finite."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (%); the divide-by-zero case collapses to a zero term."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    # Mask before dividing so a zero denominator never evaluates 0/0.
    nonzero = denom != 0
    ratio = np.zeros_like(denom)
    ratio[nonzero] = np.abs(y_true - y_pred)[nonzero] / denom[nonzero]
    return float(np.mean(2 * ratio) * 100)


def seasonal_naive_scale(train_target: np.ndarray, period: int = SEASONAL_PERIOD) -> float:
    """In-sample MAE of the lag-24 seasonal naive forecast on the training target.

    Lag-24 is chosen because the dominant cycle here is daily, so "same hour
    yesterday" is the natural no-skill reference for MASE.
    """
    y = _to_array(train_target)
    diffs = np.abs(y[period:] - y[:-period])
    return float(np.mean(diffs))


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    """Mean absolute scaled error against the precomputed seasonal naive scale."""
    if scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> dict[str, float]:
    """Return all five metrics for one model in a single dict."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, scale),
    }
