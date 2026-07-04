"""Direct next-day (h=24) XGBoost forecaster (Block B, Step 0).

Answers Business Question 1 for the next-day horizon. Reuses the tuned XGBoost
hyperparameters; the seasonal naive at h=24 is lag-24, so the existing MASE
denominator is the directly interpretable baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics, modeling, tracking

FEATURES_PATH = ROOT / "data" / "processed" / "household_power_hourly_features.parquet"


def main() -> int:
    tracking.set_seeds()
    tracking.configure()

    df = pd.read_parquet(FEATURES_PATH)
    # The next-hour matrix only supplies the training seasonal-naive MASE scale.
    scale = metrics.seasonal_naive_scale(
        modeling.build_model_matrix(df, verbose=False).y_train.to_numpy()
    )

    matrix = modeling.build_h24_matrix(df)
    print("h24 features:", matrix.features)
    print("X_train:", matrix.X_train.shape, "| X_test:", matrix.X_test.shape)

    y_pred = modeling.fit_direct_xgboost(matrix)
    scores = metrics.evaluate(matrix.y_test.to_numpy(), y_pred.to_numpy(), scale)
    pred_path = modeling.save_predictions("xgboost_h24", matrix.y_test, y_pred)
    tracking.log_run(
        "xgboost_h24",
        {"model": "xgboost_h24", "horizon": 24, **modeling.TUNED_XGB_PARAMS},
        scores,
        None,
        artifact_path=pred_path,
    )
    print("h24 test metrics:", {k: round(v, 4) for k, v in scores.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
