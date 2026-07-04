"""Train and log XGBoost and the LSTM (Phase 11c-11d).

Run inside the project virtualenv. Reuses the model matrix, metrics, MLflow
helper and the seasonal-naive MASE scale from the baseline/SARIMA stage, then
prints a unified leaderboard over every persisted model prediction.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import lstm, metrics, modeling, tracking

FEATURES_PATH = ROOT / "data" / "processed" / "household_power_hourly_features.parquet"
PREDICTIONS_DIR = ROOT / "data" / "processed" / "predictions"


def _leaderboard(scale: float) -> pd.DataFrame:
    """Recompute all five metrics for every persisted prediction file, by MAE."""
    rows = {}
    for path in sorted(PREDICTIONS_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path)
        rows[path.stem] = metrics.evaluate(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), scale
        )
    board = pd.DataFrame(rows).T[["mae", "rmse", "mape", "smape", "mase"]]
    return board.sort_values("mae")


def main() -> int:
    tracking.set_seeds()
    tracking.configure()
    tmp = Path(tempfile.mkdtemp())

    df = pd.read_parquet(FEATURES_PATH)
    matrix = modeling.build_model_matrix(df)
    scale = metrics.seasonal_naive_scale(matrix.y_train.to_numpy())

    # XGBoost (raw features, no scaling).
    print("\nfitting XGBoost...", flush=True)
    xgb = modeling.fit_xgboost(matrix, seed=tracking.SEED)
    xgb_scores = metrics.evaluate(
        matrix.y_test.to_numpy(), xgb.y_pred.to_numpy(), scale
    )
    xgb_pred_path = modeling.save_predictions("xgboost", matrix.y_test, xgb.y_pred)
    imp_path = tmp / "xgboost_importances.csv"
    xgb.importances.to_csv(imp_path, header=["importance"])
    tracking.log_run(
        "xgboost",
        {
            "model": "xgboost",
            "n_estimators": 500,
            "best_iteration": xgb.best_iteration,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        xgb_scores,
        None,
        artifact_path=xgb_pred_path,
        extra_artifacts=[imp_path],
    )
    print(f"XGBoost best_iteration={xgb.best_iteration} {xgb_scores}")
    print("top-5 importances:\n", xgb.importances.head(5).round(4).to_string())

    # LSTM (scaled target windows, CPU torch).
    print(f"\ntraining LSTM on {torch.get_num_threads()}-thread CPU torch...", flush=True)
    result = lstm.train_lstm(matrix.y_train, matrix.y_test, verbose=True)
    lstm_scores = metrics.evaluate(
        matrix.y_test.to_numpy(), result.y_pred.to_numpy(), scale
    )
    lstm_pred_path = modeling.save_predictions("lstm", matrix.y_test, result.y_pred)
    scaler_path = lstm.save_scaler(result.scaler)
    curve_path = tmp / "lstm_training_curve.csv"
    pd.Series(result.val_loss_curve, name="val_mse").to_csv(curve_path, index_label="epoch")
    tracking.log_run(
        "lstm",
        {
            "model": "lstm",
            "hidden_size": lstm.HIDDEN_SIZE,
            "num_layers": lstm.NUM_LAYERS,
            "dropout": lstm.DROPOUT,
            "lookback": lstm.LOOKBACK,
            "epochs_run": result.epochs_run,
            "train_seconds": round(result.train_seconds, 2),
        },
        lstm_scores,
        None,
        artifact_path=lstm_pred_path,
        extra_artifacts=[curve_path],
    )
    print(
        f"LSTM epochs={result.epochs_run} train={result.train_seconds:.1f}s "
        f"scaler={scaler_path.name} {lstm_scores}"
    )

    print("\n== LEADERBOARD (by MAE) ==")
    print(_leaderboard(scale).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
