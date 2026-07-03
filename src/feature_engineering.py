"""Leakage-safe feature engineering for the hourly consumption series.

Two feature families live here. Calendar, cyclical and holiday features derive
from the timestamp alone, so they are safe to compute over the whole series.
Lag and rolling features are strictly backward-looking: a test-set row may read
the tail of the training series (this mirrors real inference and is not
leakage), but no feature ever reads a future value. Every function is pure and
deterministic, so the fixed seed 42 has no effect here; fitted transforms
(scalers, imputers) are deliberately excluded and belong to the train-only
Phase 10.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "household_power_hourly_features.parquet"

TARGET = "Global_active_power"

# First timestamp of the test period; everything before it is training.
TEST_START = "2010-01-01"

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
MONTHS_PER_YEAR = 12

LAGS = [1, 24, 48, 168, 336]
ROLLING_WINDOWS = [24, 168]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar attributes; consumption tracks time-of-day and week structure."""
    out = frame.copy()
    idx = out.index
    out["hour"] = idx.hour.astype("int16")
    out["dayofweek"] = idx.dayofweek.astype("int16")
    out["day"] = idx.day.astype("int16")
    out["month"] = idx.month.astype("int16")
    out["quarter"] = idx.quarter.astype("int16")
    out["year"] = idx.year.astype("int16")
    out["is_weekend"] = (idx.dayofweek >= 5).astype("int8")
    return out


def add_cyclical_encodings(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode periodic calendar fields as sin/cos so wrap-around points stay adjacent."""
    out = frame.copy()
    idx = out.index
    hour = idx.hour.to_numpy()
    dayofweek = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()

    out["hour_sin"] = np.sin(2 * np.pi * hour / HOURS_PER_DAY)
    out["hour_cos"] = np.cos(2 * np.pi * hour / HOURS_PER_DAY)
    out["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / DAYS_PER_WEEK)
    out["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / DAYS_PER_WEEK)
    out["month_sin"] = np.sin(2 * np.pi * month / MONTHS_PER_YEAR)
    out["month_cos"] = np.cos(2 * np.pi * month / MONTHS_PER_YEAR)
    return out


def add_lag_features(
    frame: pd.DataFrame, lags: list[int] = LAGS, target: str = TARGET
) -> pd.DataFrame:
    """Add past-value lags of the target; recent demand is the strongest predictor."""
    out = frame.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(
    frame: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS,
    target: str = TARGET,
) -> pd.DataFrame:
    """Add rolling mean/std of the target, summarising recent level and volatility.

    The series is shifted by one before rolling so the current hour is never
    part of its own window, keeping the feature strictly causal.
    """
    out = frame.copy()
    shifted = out[target].shift(1)
    for window in windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()
    return out


def add_holiday_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag French public holidays; routine and demand shift on those days."""
    # Imported lazily to keep the dependency optional for callers that skip it.
    import holidays

    years = range(int(frame.index.year.min()), int(frame.index.year.max()) + 1)
    fr_holidays = holidays.France(years=years)
    out = frame.copy()
    out["is_holiday"] = (
        pd.Series(out.index.date, index=out.index).isin(fr_holidays).astype("int8")
    )
    return out


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compose every feature family on the full series (causal lags included)."""
    out = add_calendar_features(frame)
    out = add_cyclical_encodings(out)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_holiday_flag(out)
    return out


def chronological_split(frame: pd.DataFrame, test_start: str = TEST_START) -> pd.DataFrame:
    """Tag each row train/test by a fixed date; the split is never random."""
    out = frame.copy()
    boundary = pd.Timestamp(test_start)
    out["split"] = np.where(out.index < boundary, "train", "test")
    return out


def drop_warmup(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop leading rows whose lag/rolling features are undefined; report the count."""
    feature_cols = [
        col for col in frame.columns if col.startswith(("lag_", "roll_"))
    ]
    before = len(frame)
    cleaned = frame.dropna(subset=feature_cols)
    return cleaned, before - len(cleaned)


def save_features(frame: pd.DataFrame, path: Path = FEATURES_PATH) -> Path:
    """Persist the feature frame as parquet, creating the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path
