"""Leakage-safe time-series cross-validation on the training set.

Every fold uses an expanding window (TimeSeriesSplit), so validation always
sits strictly after its training slice. The test set is never touched here.
Lag/rolling features are already causal, so no per-fold recomputation is needed
for the tree and naive models; SARIMA and the LSTM are refit per fold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import lstm as lstm_mod
from . import metrics, modeling

N_SPLITS = 5
LSTM_N_SPLITS = 3  # LSTM CV is reduced: CPU cost is high and it only tests H3.
SARIMA_WINDOW = 24 * 90


@dataclass
class CVResult:
    """Per-fold validation MAE plus optional diagnostics for one model."""

    model: str
    fold_maes: list[float]
    train_maes: list[float] = field(default_factory=list)
    fit_seconds: list[float] = field(default_factory=list)

    @property
    def mean_mae(self) -> float:
        return float(np.mean(self.fold_maes))

    @property
    def std_mae(self) -> float:
        return float(np.std(self.fold_maes))

    @property
    def gap(self) -> float | None:
        """Mean CV-minus-train MAE gap; positive means overfitting."""
        if not self.train_maes:
            return None
        return self.mean_mae - float(np.mean(self.train_maes))


def _splits(n: int, n_splits: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def cv_naive(matrix: modeling.ModelMatrix, lag: int, n_splits: int = N_SPLITS) -> CVResult:
    """CV MAE for a seasonal naive forecast (the lag column on each val fold)."""
    y = matrix.y_train.to_numpy()
    preds = matrix.X_train[f"lag_{lag}"].to_numpy()
    maes = []
    for _, val_idx in _splits(len(y), n_splits).split(y):
        maes.append(metrics.mae(y[val_idx], preds[val_idx]))
    return CVResult(f"naive_lag{lag}", maes)


def cv_xgboost(matrix: modeling.ModelMatrix, n_splits: int = N_SPLITS) -> CVResult:
    """CV MAE for XGBoost, with per-fold train MAE to size the overfitting gap."""
    from xgboost import XGBRegressor

    x = matrix.X_train
    y = matrix.y_train
    val_maes, train_maes = [], []

    for train_idx, val_idx in _splits(len(y), n_splits).split(x):
        x_tr, y_tr = x.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
        # Match the first-pass fixed params; no early stopping so folds are comparable.
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_tr, y_tr)
        val_maes.append(metrics.mae(y_val.to_numpy(), model.predict(x_val)))
        train_maes.append(metrics.mae(y_tr.to_numpy(), model.predict(x_tr)))

    return CVResult("xgboost", val_maes, train_maes=train_maes)


def cv_sarima(
    matrix: modeling.ModelMatrix,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    window: int = SARIMA_WINDOW,
    n_splits: int = N_SPLITS,
) -> CVResult:
    """CV MAE for SARIMA, refitting the recent-window model on each fold."""
    y = matrix.y_train
    maes, times = [], []

    for train_idx, val_idx in _splits(len(y), n_splits).split(y):
        fold_train = y.iloc[train_idx]
        fold_val = y.iloc[val_idx]
        fit = modeling.fit_sarima(
            fold_train, order, seasonal_order, train_window=min(window, len(fold_train))
        )
        pred = modeling.sarima_one_step(fit, fold_val)
        maes.append(metrics.mae(fold_val.to_numpy(), pred.to_numpy()))
        times.append(fit.fit_seconds)

    return CVResult("sarima", maes, fit_seconds=times)


def cv_lstm(matrix: modeling.ModelMatrix, n_splits: int = LSTM_N_SPLITS) -> CVResult:
    """CV MAE for the LSTM, refit per fold with early stopping to stay tractable."""
    y = matrix.y_train
    maes, times = [], []

    for train_idx, val_idx in _splits(len(y), n_splits).split(y):
        fold_train = y.iloc[train_idx]
        fold_val = y.iloc[val_idx]
        start = time.perf_counter()
        result = lstm_mod.train_lstm(fold_train, fold_val)
        times.append(time.perf_counter() - start)
        maes.append(metrics.mae(fold_val.to_numpy(), result.y_pred.to_numpy()))

    return CVResult("lstm", maes, fit_seconds=times)
