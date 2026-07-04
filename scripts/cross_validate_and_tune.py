"""Time-series cross-validation and XGBoost tuning (Phase 12-13).

Run inside the project virtualenv. Cross-validates every model on the training
set with an expanding window, tunes XGBoost with a randomized search, and logs
all results to MLflow. The test set is only touched to score the final models.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cross_validation as cv
from src import metrics, modeling, tracking

FEATURES_PATH = ROOT / "data" / "processed" / "household_power_hourly_features.parquet"
PREDICTIONS_DIR = ROOT / "data" / "processed" / "predictions"

SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)


def _log_cv(result: cv.CVResult) -> None:
    """Log one model's CV summary as an MLflow run."""
    import mlflow

    metrics_out = {"cv_mae_mean": result.mean_mae, "cv_mae_std": result.std_mae}
    if result.train_maes:
        metrics_out["cv_train_mae_mean"] = float(np.mean(result.train_maes))
        metrics_out["cv_gap"] = result.gap
    if result.fit_seconds:
        metrics_out["fold_fit_seconds_mean"] = float(np.mean(result.fit_seconds))
    with mlflow.start_run(run_name=f"cv_{result.model}"):
        mlflow.log_param("n_splits", len(result.fold_maes))
        mlflow.log_metrics(metrics_out)


def main() -> int:
    tracking.set_seeds()
    tracking.configure()

    df = pd.read_parquet(FEATURES_PATH)
    matrix = modeling.build_model_matrix(df)
    scale = metrics.seasonal_naive_scale(matrix.y_train.to_numpy())

    # -- Phase 12: cross-validation --
    print("\n== PHASE 12: CROSS-VALIDATION ==", flush=True)
    results: list[cv.CVResult] = []
    for lag in (1, 24, 168):
        results.append(cv.cv_naive(matrix, lag))
    print("naive folds done", flush=True)
    results.append(cv.cv_xgboost(matrix))
    print("xgboost CV done", flush=True)
    results.append(cv.cv_sarima(matrix, SARIMA_ORDER, SARIMA_SEASONAL_ORDER))
    print("sarima CV done", flush=True)
    results.append(cv.cv_lstm(matrix))
    print("lstm CV done", flush=True)

    for result in results:
        _log_cv(result)

    rows = {
        r.model: {
            "cv_mae_mean": r.mean_mae,
            "cv_mae_std": r.std_mae,
            "train_vs_cv_gap": r.gap,
        }
        for r in results
    }
    cv_table = pd.DataFrame(rows).T.sort_values("cv_mae_mean")
    print("\n-- CV SUMMARY (by mean MAE) --")
    print(cv_table.round(4).to_string())
    xgb_cv = next(r for r in results if r.model == "xgboost")

    # -- Phase 13: XGBoost tuning --
    print("\n== PHASE 13: XGBOOST TUNING ==", flush=True)
    t0 = time.perf_counter()
    tuned = modeling.tune_xgboost(matrix)
    print(f"randomized search done in {time.perf_counter() - t0:.1f}s", flush=True)

    tuned_scores = metrics.evaluate(
        matrix.y_test.to_numpy(), tuned.y_pred.to_numpy(), scale
    )
    tuned_pred_path = modeling.save_predictions(
        "xgboost_tuned", matrix.y_test, tuned.y_pred
    )
    tracking.log_run(
        "xgboost_tuned",
        {"model": "xgboost_tuned", **{k: round(float(v), 4) if isinstance(v, float) else v
                                      for k, v in tuned.best_params.items()},
         "cv_mae": round(tuned.cv_mae, 4)},
        tuned_scores,
        None,
        artifact_path=tuned_pred_path,
    )

    untuned_test = metrics.evaluate(
        *(
            (lambda f: (f["y_true"].to_numpy(), f["y_pred"].to_numpy()))(
                pd.read_parquet(PREDICTIONS_DIR / "xgboost.parquet")
            )
        ),
        scale,
    )

    print("\n-- TUNED vs UNTUNED XGBoost --")
    print(f"best params: {tuned.best_params}")
    print(f"untuned CV MAE: {xgb_cv.mean_mae:.4f} | tuned CV MAE: {tuned.cv_mae:.4f}")
    winner = "tuned" if tuned.cv_mae < xgb_cv.mean_mae else "untuned"
    print(f"winner on CV MAE: {winner}")
    print(f"untuned TEST: {untuned_test}")
    print(f"tuned   TEST: {tuned_scores}")

    # -- Updated leaderboard on test MAE --
    board = {}
    for path in sorted(PREDICTIONS_DIR.glob("*.parquet")):
        f = pd.read_parquet(path)
        board[path.stem] = metrics.evaluate(
            f["y_true"].to_numpy(), f["y_pred"].to_numpy(), scale
        )
    leaderboard = pd.DataFrame(board).T[["mae", "rmse", "mape", "smape", "mase"]]
    print("\n== LEADERBOARD (test, by MAE) ==")
    print(leaderboard.sort_values("mae").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
