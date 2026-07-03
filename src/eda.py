"""Exploratory plots for the hourly consumption series.

Each function writes one figure and returns its path so a notebook or script
can drive the full set. Plots target ``Global_active_power`` in kWh, the
forecasting objective.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # File output only; no interactive backend required.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"

TARGET = "Global_active_power"
DPI = 120
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, name: str, figure_dir: Path) -> Path:
    """Save and close a figure, returning its path."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_full_series(series: pd.Series, figure_dir: Path = FIGURE_DIR) -> Path:
    """Plot the entire series to expose trend and seasonal envelope."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(series.index, series.to_numpy(), linewidth=0.4)
    ax.set_title("Hourly global active power (full history)")
    ax.set_ylabel("kWh")
    return _save(fig, "01_full_series.png", figure_dir)


def plot_zoom(series: pd.Series, figure_dir: Path = FIGURE_DIR) -> Path:
    """Plot one representative week and one day to reveal fine structure."""
    start = series.index.min() + pd.Timedelta(days=30)
    week = series.loc[start : start + pd.Timedelta(days=7)]
    day = series.loc[start : start + pd.Timedelta(days=1)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(week.index, week.to_numpy(), linewidth=0.8)
    axes[0].set_title("One week")
    axes[0].set_ylabel("kWh")
    axes[1].plot(day.index, day.to_numpy(), marker="o", markersize=2, linewidth=0.8)
    axes[1].set_title("One day")
    axes[1].set_ylabel("kWh")
    fig.tight_layout()
    return _save(fig, "02_zoom_week_day.png", figure_dir)


def plot_boxplots(frame: pd.DataFrame, figure_dir: Path = FIGURE_DIR) -> Path:
    """Boxplots of the target grouped by hour, day-of-week and month."""
    idx = frame.index
    tmp = pd.DataFrame(
        {
            "value": frame[TARGET].to_numpy(),
            "hour": idx.hour,
            "day_of_week": idx.dayofweek,
            "month": idx.month,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    sns.boxplot(data=tmp, x="hour", y="value", ax=axes[0], fliersize=0.5)
    axes[0].set_title("Distribution by hour of day")
    sns.boxplot(data=tmp, x="day_of_week", y="value", ax=axes[1], fliersize=0.5)
    axes[1].set_title("Distribution by day of week (0=Mon)")
    sns.boxplot(data=tmp, x="month", y="value", ax=axes[2], fliersize=0.5)
    axes[2].set_title("Distribution by month")
    for ax in axes:
        ax.set_ylabel("kWh")
    fig.tight_layout()
    return _save(fig, "03_boxplots.png", figure_dir)


def plot_rolling_stats(series: pd.Series, figure_dir: Path = FIGURE_DIR) -> Path:
    """Overlay daily and weekly rolling means with a weekly std band."""
    mean_24 = series.rolling(DAILY_PERIOD).mean()
    mean_168 = series.rolling(WEEKLY_PERIOD).mean()
    std_168 = series.rolling(WEEKLY_PERIOD).std()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(series.index, series.to_numpy(), linewidth=0.2, alpha=0.4, label="hourly")
    ax.plot(mean_24.index, mean_24.to_numpy(), linewidth=0.8, label="24h mean")
    ax.plot(mean_168.index, mean_168.to_numpy(), linewidth=1.0, label="168h mean")
    ax.fill_between(
        mean_168.index,
        (mean_168 - std_168).to_numpy(),
        (mean_168 + std_168).to_numpy(),
        alpha=0.2,
        label="168h +/- 1 std",
    )
    ax.set_title("Rolling statistics")
    ax.set_ylabel("kWh")
    ax.legend(loc="upper right")
    return _save(fig, "04_rolling_stats.png", figure_dir)


def plot_acf_pacf(
    series: pd.Series, lags: int = WEEKLY_PERIOD, figure_dir: Path = FIGURE_DIR
) -> Path:
    """Plot ACF and PACF up to ``lags`` to expose autocorrelation structure."""
    clean = series.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    plot_acf(clean, lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF (up to {lags} lags)")
    plot_pacf(clean, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF (up to {lags} lags)")
    fig.tight_layout()
    return _save(fig, "05_acf_pacf.png", figure_dir)


def plot_stl(
    series: pd.Series, period: int, name: str, figure_dir: Path = FIGURE_DIR
) -> Path:
    """Run and plot an STL decomposition at the given seasonal period."""
    result = STL(series.dropna(), period=period, robust=True).fit()
    fig = result.plot()
    fig.set_size_inches(14, 9)
    fig.suptitle(f"STL decomposition (period={period})")
    return _save(fig, name, figure_dir)


def plot_distribution(series: pd.Series, figure_dir: Path = FIGURE_DIR) -> Path:
    """Histogram on linear and log axes to expose the heavy right tail."""
    values = series.dropna().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(values, bins=80, ax=axes[0])
    axes[0].set_title("Distribution (linear)")
    axes[0].set_xlabel("kWh")
    sns.histplot(values, bins=80, ax=axes[1], log_scale=(False, True))
    axes[1].set_title("Distribution (log count)")
    axes[1].set_xlabel("kWh")
    fig.tight_layout()
    return _save(fig, "08_distribution.png", figure_dir)


def plot_load_heatmap(frame: pd.DataFrame, figure_dir: Path = FIGURE_DIR) -> Path:
    """Mean load over the hour-by-day-of-week grid: the weekly load profile."""
    idx = frame.index
    tmp = pd.DataFrame(
        {
            "value": frame[TARGET].to_numpy(),
            "hour": idx.hour,
            "day_of_week": idx.dayofweek,
        }
    )
    pivot = tmp.pivot_table(index="hour", columns="day_of_week", values="value")
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(pivot, cmap="viridis", ax=ax, cbar_kws={"label": "mean kWh"})
    ax.set_title("Mean load by hour and day of week (0=Mon)")
    return _save(fig, "09_load_heatmap.png", figure_dir)


def plot_monthly_profile(series: pd.Series, figure_dir: Path = FIGURE_DIR) -> Path:
    """Monthly mean over time to expose annual seasonality and drift."""
    monthly = series.resample("MS").mean()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly.index, monthly.to_numpy(), marker="o", markersize=3)
    ax.set_title("Monthly mean consumption")
    ax.set_ylabel("kWh")
    return _save(fig, "10_monthly_profile.png", figure_dir)


def generate_all(frame: pd.DataFrame, figure_dir: Path = FIGURE_DIR) -> list[Path]:
    """Produce the full EDA figure set and return the written paths."""
    series = frame[TARGET]
    paths = [
        plot_full_series(series, figure_dir),
        plot_zoom(series, figure_dir),
        plot_boxplots(frame, figure_dir),
        plot_rolling_stats(series, figure_dir),
        plot_acf_pacf(series, figure_dir=figure_dir),
        plot_stl(series, DAILY_PERIOD, "06_stl_daily.png", figure_dir),
        plot_stl(series, WEEKLY_PERIOD, "07_stl_weekly.png", figure_dir),
        plot_distribution(series, figure_dir),
        plot_load_heatmap(frame, figure_dir),
        plot_monthly_profile(series, figure_dir),
    ]
    return paths
