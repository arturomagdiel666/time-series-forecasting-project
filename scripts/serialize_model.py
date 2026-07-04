"""Serialize the winning models and their metadata (Block C, Step 2).

Persists the tuned next-hour XGBoost and the direct next-day (h=24) XGBoost, plus
a metadata file the dashboard reads for feature bounds, hyperparameters and the
reported test metrics. Run inside the project virtualenv.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics, modeling

FEATURES_PATH = ROOT / "data" / "processed" / "household_power_hourly_features.parquet"
PREDICTIONS_DIR = ROOT / "data" / "processed" / "predictions"


def _fit_tuned(matrix: modeling.ModelMatrix) -> object:
    """Fit XGBoost with the tuned params; matches the persisted xgboost_tuned run."""
    from xgboost import XGBRegressor

    # n_jobs=1 mirrors the RandomizedSearchCV base estimator so predictions
    # reproduce the stored xgboost_tuned.parquet exactly.
    model = XGBRegressor(random_state=42, n_jobs=1, **modeling.TUNED_XGB_PARAMS)
    model.fit(matrix.X_train, matrix.y_train)
    return model


def _metrics_from(name: str, scale: float) -> dict[str, float]:
    frame = pd.read_parquet(PREDICTIONS_DIR / f"{name}.parquet")
    return metrics.evaluate(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), scale)


def main() -> int:
    df = pd.read_parquet(FEATURES_PATH)
    matrix = modeling.build_model_matrix(df, verbose=False)
    matrix_h24 = modeling.build_h24_matrix(df)
    scale = metrics.seasonal_naive_scale(matrix.y_train.to_numpy())

    modeling.MODELS_DIR.mkdir(exist_ok=True)

    from xgboost import XGBRegressor

    model_h1 = _fit_tuned(matrix)
    model_h24 = XGBRegressor(random_state=42, n_jobs=1, **modeling.TUNED_XGB_PARAMS)
    model_h24.fit(matrix_h24.X_train, matrix_h24.y_train)

    joblib.dump(model_h1, modeling.BEST_MODEL_PATH)
    joblib.dump(model_h24, modeling.BEST_MODEL_H24_PATH)

    # Per-feature train statistics give the dashboard sensible input bounds.
    bounds = {
        col: {
            "min": float(matrix.X_train[col].min()),
            "max": float(matrix.X_train[col].max()),
            "mean": float(matrix.X_train[col].mean()),
        }
        for col in matrix.features
    }

    metadata = {
        "target": "Global_active_power",
        "units": "kWh",
        "features": matrix.features,
        "feature_bounds": bounds,
        "tuned_hyperparameters": modeling.TUNED_XGB_PARAMS,
        "metrics": {
            "h1": _metrics_from("xgboost_tuned", scale),
            "h24": _metrics_from("xgboost_h24", scale),
        },
        "mase_denominator": round(float(scale), 4),
        "seed": 42,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    modeling.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Sanity check: reloaded model must reproduce the stored tuned predictions.
    reloaded = modeling.load_model()
    stored = pd.read_parquet(PREDICTIONS_DIR / "xgboost_tuned.parquet")
    sample = matrix.X_test.iloc[:5]
    reloaded_pred = reloaded.predict(sample)
    stored_pred = stored.loc[sample.index, "y_pred"].to_numpy()
    match = np.allclose(reloaded_pred, stored_pred, rtol=1e-4, atol=1e-4)

    print("saved:", modeling.BEST_MODEL_PATH.name, modeling.BEST_MODEL_H24_PATH.name,
          modeling.METADATA_PATH.name)
    print("h1 metrics:", {k: round(v, 4) for k, v in metadata["metrics"]["h1"].items()})
    print("h24 metrics:", {k: round(v, 4) for k, v in metadata["metrics"]["h24"].items()})
    print("reload sanity (5 rows) matches stored xgboost_tuned:", match)
    print("  reloaded:", np.round(reloaded_pred, 5).tolist())
    print("  stored  :", np.round(stored_pred, 5).tolist())
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
