"""MLflow tracking helpers.

Centralises the local tracking location, experiment name and seeding so every
model run is logged the same way and stays reproducible.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "household-power-forecasting"
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and torch (if present) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def configure() -> None:
    """Point MLflow at the local ``mlruns/`` store and select the experiment."""
    # MLflow 3.x gates the file-store backend; opt in since the project pins a
    # local mlruns/ directory rather than a database backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    TRACKING_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(
    model_name: str,
    params: dict[str, object],
    metrics: dict[str, float],
    predictions: pd.DataFrame,
    artifact_path: Path | None = None,
) -> str:
    """Log one model run: params, all five metrics and the prediction artifact."""
    import mlflow

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("seed", SEED)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifact_path is not None and artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="predictions")
        else:
            # Fall back to logging the frame inline when no file is provided.
            mlflow.log_text(predictions.to_csv(), f"{model_name}_predictions.csv")
        return run.info.run_id
