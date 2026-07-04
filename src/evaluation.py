"""Evaluation, residual diagnostics and error-analysis plots (Phase 14-15).

Reads the persisted per-model predictions and turns them into the comparison
table and figures used for model selection and interpretation. Figures are
written at 120 dpi to stay consistent with the Block A EDA output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from . import metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "modeling"
DPI = 120

# Every persisted model is a next-hour forecaster except the direct h=24 model.
HORIZONS = {
    "naive_lag1": "h=1",
    "naive_lag24": "h=1",
    "naive_lag168": "h=1",
    "sarima": "h=1",
    "lstm": "h=1",
    "xgboost": "h=1",
    "xgboost_tuned": "h=1",
    "xgboost_h24": "h=24",
}

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, name: str, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def load_predictions(pred_dir: Path = PREDICTIONS_DIR) -> dict[str, pd.DataFrame]:
    """Load every persisted prediction frame keyed by model name."""
    return {p.stem: pd.read_parquet(p) for p in sorted(pred_dir.glob("*.parquet"))}


def metrics_table(preds: dict[str, pd.DataFrame], scale: float) -> pd.DataFrame:
    """Assemble the labelled test metrics table, sorted by MAE within horizon."""
    rows = {}
    for name, frame in preds.items():
        scores = metrics.evaluate(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), scale
        )
        scores["horizon"] = HORIZONS.get(name, "h=1")
        rows[name] = scores
    table = pd.DataFrame(rows).T[["horizon", "mae", "rmse", "mape", "smape", "mase"]]
    return table.sort_values(["horizon", "mae"])


def plot_actual_vs_pred_week(
    preds: dict[str, pd.DataFrame],
    models: list[str],
    week_start: str,
    figure_dir: Path = FIGURE_DIR,
) -> Path:
    """Overlay actuals and model predictions across one representative week."""
    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=7)
    ref = preds[models[0]]
    window = ref.loc[start:end]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(window.index, window["y_true"].to_numpy(), color="black", linewidth=1.5, label="actual")
    for name in models:
        seg = preds[name].loc[start:end]
        ax.plot(seg.index, seg["y_pred"].to_numpy(), linewidth=1.0, alpha=0.8, label=name)
    ax.set_title(f"Actual vs predicted, week of {start.date()}")
    ax.set_ylabel("kWh")
    ax.legend(loc="upper right")
    return _save(fig, "01_actual_vs_pred_week.png", figure_dir)


def residual_diagnostics(
    name: str, frame: pd.DataFrame, figure_dir: Path = FIGURE_DIR
) -> Path:
    """Residual histogram, ACF and Q-Q plot for one model."""
    resid = (frame["y_true"] - frame["y_pred"]).dropna()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(resid.to_numpy(), bins=60, ax=axes[0])
    axes[0].set_title(f"{name}: residual histogram")
    axes[0].set_xlabel("residual (kWh)")
    plot_acf(resid, lags=48, ax=axes[1])
    axes[1].set_title(f"{name}: residual ACF (48 lags)")
    stats.probplot(resid.to_numpy(), dist="norm", plot=axes[2])
    axes[2].set_title(f"{name}: Q-Q plot")
    fig.tight_layout()
    return _save(fig, f"02_residuals_{name}.png", figure_dir)


def mae_by(frame: pd.DataFrame, attribute: str) -> pd.Series:
    """Mean absolute error grouped by a calendar attribute of the timestamp."""
    idx = frame.index
    key = {"hour": idx.hour, "dayofweek": idx.dayofweek, "month": idx.month}[attribute]
    abs_err = (frame["y_true"] - frame["y_pred"]).abs()
    return abs_err.groupby(key).mean()


def plot_error_breakdowns(
    preds: dict[str, pd.DataFrame],
    models: list[str],
    figure_dir: Path = FIGURE_DIR,
) -> Path:
    """MAE by hour, day-of-week and month for the compared models."""
    attributes = ["hour", "dayofweek", "month"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, attr in zip(axes, attributes):
        for name in models:
            series = mae_by(preds[name], attr)
            ax.plot(series.index, series.to_numpy(), marker="o", markersize=3, label=name)
        ax.set_title(f"MAE by {attr}")
        ax.set_xlabel(attr)
        ax.set_ylabel("MAE (kWh)")
        ax.legend()
    fig.tight_layout()
    return _save(fig, "03_error_breakdowns.png", figure_dir)


def plot_feature_importances(
    importances: pd.Series, figure_dir: Path = FIGURE_DIR, top_n: int = 15
) -> Path:
    """Horizontal bar chart of the top feature importances."""
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.to_numpy())
    ax.set_title("XGBoost feature importances (top 15)")
    ax.set_xlabel("gain importance")
    fig.tight_layout()
    return _save(fig, "04_feature_importances.png", figure_dir)
