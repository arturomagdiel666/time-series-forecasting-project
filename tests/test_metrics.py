"""Unit tests for the forecast error metrics."""

from __future__ import annotations

import numpy as np

from src import metrics


def test_mae_and_rmse_hand_computed() -> None:
    """MAE and RMSE match values computed by hand."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])  # errors: 0, 0, -2
    assert np.isclose(metrics.mae(y_true, y_pred), 2 / 3)
    assert np.isclose(metrics.rmse(y_true, y_pred), np.sqrt(4 / 3))


def test_mape_smape_are_zero_guarded() -> None:
    """Zero actuals must not produce inf or NaN in MAPE/sMAPE."""
    y_true = np.array([0.0, 10.0, 0.0, 20.0])
    y_pred = np.array([1.0, 11.0, 2.0, 18.0])
    assert np.isfinite(metrics.mape(y_true, y_pred))
    assert np.isfinite(metrics.smape(y_true, y_pred))


def test_smape_all_zero_is_finite() -> None:
    """The degenerate all-zero case collapses to zero rather than NaN."""
    zeros = np.zeros(4)
    assert metrics.smape(zeros, zeros) == 0.0


def test_seasonal_naive_scale_hand_computed() -> None:
    """The scale equals the mean absolute lag-period difference of the train series."""
    train = np.array([1.0, 3.0, 2.0, 6.0])  # period-2 diffs: |2-1|=1, |6-3|=3
    assert np.isclose(metrics.seasonal_naive_scale(train, period=2), 2.0)


def test_mase_equals_mae_over_scale() -> None:
    """MASE is MAE divided by the supplied seasonal-naive scale."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    scale = 0.5
    assert np.isclose(
        metrics.mase(y_true, y_pred, scale), metrics.mae(y_true, y_pred) / scale
    )


def test_mase_below_one_for_better_than_naive() -> None:
    """A model beating the naive reference scores MASE < 1."""
    train = np.array([1.0, 3.0, 2.0, 6.0, 1.0, 3.0])
    scale = metrics.seasonal_naive_scale(train, period=2)
    y_true = np.array([2.0, 4.0, 3.0])
    y_pred = y_true + 0.05  # near-perfect, well under the naive error
    assert metrics.mase(y_true, y_pred, scale) < 1.0
