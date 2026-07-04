"""Train and log the naive baselines and SARIMA (Phase 11a-11b).

Run inside the project virtualenv. Logs every run to the local MLflow store and
writes per-model test predictions for later aggregation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import metrics, modeling, tracking

FEATURES_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "household_power_hourly_features.parquet"
)

# Orders informed by Block A: D=1 at s=24 (H2: stationary after lag-24 seasonal
# differencing); d=0 since the raw level is already mean-reverting (ADF); p=q=1
# for short-run intraday autocorrelation; P=Q=1 for the daily seasonal structure.
SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)
SARIMA_TRAIN_WINDOW = 24 * 90  # last ~90 days, contiguous with the test start.


def _run_and_log(
    name: str,
    params: dict[str, object],
    y_true: pd.Series,
    y_pred: pd.Series,
    scale: float,
) -> dict[str, float]:
    """Evaluate, persist predictions and log one MLflow run."""
    scores = metrics.evaluate(y_true.to_numpy(), y_pred.to_numpy(), scale)
    pred_path = modeling.save_predictions(name, y_true, y_pred)
    run_id = tracking.log_run(name, params, scores, None, artifact_path=pred_path)
    print(f"[{name}] run={run_id[:8]} {scores}")
    return scores


def main() -> int:
    tracking.set_seeds()
    tracking.configure()

    df = pd.read_parquet(FEATURES_PATH)
    matrix = modeling.build_model_matrix(df)
    scale = metrics.seasonal_naive_scale(matrix.y_train.to_numpy())
    print(f"seasonal-naive (lag-24) MASE scale: {scale:.4f}")

    results: dict[str, dict[str, float]] = {}

    # Phase 11a: naive baselines.
    for lag, name in [(1, "naive_lag1"), (24, "naive_lag24"), (168, "naive_lag168")]:
        y_pred = modeling.naive_forecast(matrix, lag)
        results[name] = _run_and_log(
            name, {"model": "naive", "lag": lag}, matrix.y_test, y_pred, scale
        )

    # Phase 11b: SARIMA.
    print("\nfitting SARIMA...", flush=True)
    fit = modeling.fit_sarima(
        matrix.y_train,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        train_window=SARIMA_TRAIN_WINDOW,
    )
    print(f"SARIMA fit: window={fit.train_window} hours, {fit.fit_seconds:.1f}s")
    y_pred = modeling.sarima_one_step(fit, matrix.y_test)
    results["sarima"] = _run_and_log(
        "sarima",
        {
            "model": "sarima",
            "order": str(fit.order),
            "seasonal_order": str(fit.seasonal_order),
            "train_window": fit.train_window,
            "fit_seconds": round(fit.fit_seconds, 2),
        },
        matrix.y_test,
        y_pred,
        scale,
    )

    print("\n== RESULTS ==")
    table = pd.DataFrame(results).T[["mae", "rmse", "mape", "smape", "mase"]]
    print(table.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
