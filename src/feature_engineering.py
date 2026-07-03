"""Calendar and cyclical feature construction.

Only leakage-free features live here: every value derives from the timestamp
alone. Lag and rolling features are deferred to after the chronological split
so no future information reaches the training window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "household_power_hourly_features.parquet"

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar attributes derived from the DatetimeIndex."""
    out = frame.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype("int8")
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["year"] = idx.year
    out["week_of_year"] = idx.isocalendar().week.astype("int32").to_numpy()
    out["day_of_year"] = idx.dayofyear
    return out


def add_holiday_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag French public holidays; behaviour on those days departs from routine."""
    # Imported lazily to keep the dependency optional for callers that skip it.
    import holidays

    years = range(int(frame.index.year.min()), int(frame.index.year.max()) + 1)
    fr_holidays = holidays.France(years=years)
    out = frame.copy()
    out["is_holiday"] = pd.Series(out.index.date, index=out.index).isin(
        fr_holidays
    ).astype("int8")
    return out


def add_cyclical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode hour and day-of-week as sin/cos so 23h and 0h sit adjacent."""
    out = frame.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / HOURS_PER_DAY)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / HOURS_PER_DAY)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / DAYS_PER_WEEK)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / DAYS_PER_WEEK)
    return out


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the full leakage-free feature pipeline."""
    out = add_calendar_features(frame)
    out = add_holiday_flag(out)
    out = add_cyclical_features(out)
    return out


def save_features(frame: pd.DataFrame, path: Path = FEATURES_PATH) -> Path:
    """Persist the feature frame as parquet, creating the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path
