"""Run the zero-shot TimesFM 2.5 baseline: forecast, score, log MLflow, save parquets.

HEAVY + OPTIONAL. Requires ``pip install -r requirements-timesfm.txt`` and downloads
the ~1GB ``google/timesfm-2.5-200m-pytorch`` checkpoint to the HuggingFace cache
(outside the repo). This script is NOT part of CI or the Railway deploy; run it once
to (re)generate the committed TimesFM predictions that the dashboard and report read.

    python scripts/run_timesfm.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics, timesfm_forecaster as tfm, tracking  # noqa: E402

TARGET = "Global_active_power"
# Fixed project-wide seasonal-naive scale (models/metadata.json); identical to
# what the dashboard/report use, so TimesFM's MASE is comparable to every model.
MASE_DENOMINATOR = 0.6551
HOURLY_PATH = ROOT / "data" / "processed" / "household_power_hourly.parquet"
PRED_DIR = ROOT / "data" / "processed" / "predictions"
# Canonical test index + y_true: reuse an existing prediction file so TimesFM is
# scored on exactly the same 7,918 timestamps as every other model.
REFERENCE_PRED = PRED_DIR / "xgboost_tuned.parquet"

# (horizon, output name) — h=24 aligns to the same targets as xgboost_h24.
HORIZONS = [(1, "timesfm"), (24, "timesfm_h24")]


def main() -> int:
    series = pd.read_parquet(HOURLY_PATH)[TARGET]
    reference = pd.read_parquet(REFERENCE_PRED)
    test_index = reference.index
    y_true = series.reindex(test_index)

    # Sanity: our raw-series targets must match the shared y_true exactly.
    assert (y_true.to_numpy() == reference["y_true"].to_numpy()).all(), "y_true mismatch"

    # Load the model lazily and only if some horizon still needs forecasting, so a
    # re-run that just wants to (re)log MLflow from the committed parquets is cheap.
    model = None
    scored: dict[str, dict[str, float]] = {}
    timings: dict[str, float] = {}
    for horizon, name in HORIZONS:
        out_path = PRED_DIR / f"{name}.parquet"
        if out_path.exists():
            out = pd.read_parquet(out_path)
            timings[name] = 0.0
            reused = " (reused committed parquet)"
        else:
            if model is None:
                print(f"Loading TimesFM {tfm.CHECKPOINT} (CPU)…")
                model = tfm.load_model()
            t0 = time.perf_counter()
            y_pred = tfm.rolling_forecast(series, test_index, horizon, model=model)
            timings[name] = time.perf_counter() - t0
            out = pd.DataFrame(
                {"y_true": y_true.to_numpy(), "y_pred": y_pred.to_numpy()},
                index=test_index,
            )
            out.index.name = "timestamp"
            out.to_parquet(out_path)
            reused = f" ({timings[name] / 60:.1f} min)"

        scores = metrics.evaluate(
            out["y_true"].to_numpy(), out["y_pred"].to_numpy(), MASE_DENOMINATOR
        )
        scored[name] = scores
        print(f"{name:>12} (h={horizon:>2}){reused}: "
              + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    # MLflow: one run for the zero-shot baseline, both horizons logged.
    # tracking.configure() opts into the file store (MLFLOW_ALLOW_FILE_STORE) and
    # selects the shared experiment, matching the rest of the project's runs.
    tracking.configure()
    import mlflow

    with mlflow.start_run(run_name="timesfm_zeroshot"):
        mlflow.log_params(
            {
                "model": "timesfm-2.5-200m-pytorch",
                "category": "zero-shot foundation model",
                "context_len": tfm.CONTEXT_LEN,
                "horizons": "1,24",
                "normalize_inputs": True,
                "seed": tfm.SEED,
            }
        )
        for name, scores in scored.items():
            for metric_name, value in scores.items():
                mlflow.log_metric(f"{name}_{metric_name}", value)
            mlflow.log_metric(f"{name}_infer_seconds", timings[name])

    total = sum(timings.values())
    print(f"\nDone in {total / 60:.1f} min. Wrote timesfm.parquet + timesfm_h24.parquet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
