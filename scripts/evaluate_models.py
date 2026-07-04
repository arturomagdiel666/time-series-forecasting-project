"""Evaluation, error analysis and interpretation figures (Phase 14-16).

Run inside the project virtualenv. Produces the labelled metrics table, the
representative-week overlay, residual diagnostics, error breakdowns and the
feature-importance recap, then prints the numbers used for interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluation, metrics, modeling

FEATURES_PATH = ROOT / "data" / "processed" / "household_power_hourly_features.parquet"

# Highest-demand full week in the test set (peak winter heating, high variance);
# the hardest regime, so the clearest place to contrast model behaviour.
REPRESENTATIVE_WEEK = "2010-02-08"
LEAD_MODELS = ["xgboost_tuned", "sarima", "lstm"]


def main() -> int:
    df = pd.read_parquet(FEATURES_PATH)
    matrix = modeling.build_model_matrix(df, verbose=False)
    scale = metrics.seasonal_naive_scale(matrix.y_train.to_numpy())

    preds = evaluation.load_predictions()

    print("== PHASE 14: TEST METRICS (labelled by horizon) ==")
    table = evaluation.metrics_table(preds, scale)
    print(table.round(4).to_string())

    figs = [evaluation.plot_actual_vs_pred_week(preds, LEAD_MODELS, REPRESENTATIVE_WEEK)]
    for name in LEAD_MODELS:
        figs.append(evaluation.residual_diagnostics(name, preds[name]))
    figs.append(evaluation.plot_error_breakdowns(preds, ["xgboost_tuned", "sarima"]))

    # Feature importances of the tuned next-hour model (the winner).
    from xgboost import XGBRegressor

    winner = XGBRegressor(random_state=42, n_jobs=-1, **modeling.TUNED_XGB_PARAMS)
    winner.fit(matrix.X_train, matrix.y_train)
    importances = pd.Series(
        winner.feature_importances_, index=matrix.features
    ).sort_values(ascending=False)
    figs.append(evaluation.plot_feature_importances(importances))

    print("\nfigures written:")
    for path in figs:
        print("  ", path.name, path.exists())

    print("\n== PHASE 15: ERROR BREAKDOWN (MAE) ==")
    for attr in ("hour", "dayofweek", "month"):
        print(f"\n-- MAE by {attr} --")
        combined = pd.DataFrame(
            {
                "xgboost_tuned": evaluation.mae_by(preds["xgboost_tuned"], attr),
                "sarima": evaluation.mae_by(preds["sarima"], attr),
            }
        )
        print(combined.round(4).to_string())

    print("\ntop-10 feature importances:")
    print(importances.head(10).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
