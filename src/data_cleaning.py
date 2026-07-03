"""Cleaning and hourly aggregation of the minute-level series.

Gap handling depends on gap length: short outages carry the last reading
forward (consumption rarely jumps within an hour), while long outages get
time interpolation to avoid flat-lining a whole period. Aggregation converts
kW minute samples into kWh per hour, the unit the whole project forecasts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FFILL_LIMIT_MINUTES = 60
MINUTE_FREQ = "1min"
HOURLY_FREQ = "h"
KW_MINUTE_TO_KWH = 1.0 / 60.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HOURLY_PATH = PROCESSED_DIR / "household_power_hourly.parquet"

ENERGY_COLUMNS = ["Global_active_power", "Global_reactive_power"]
MEAN_COLUMNS = ["Voltage", "Global_intensity"]
SUB_METERING_COLUMNS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def to_regular_grid(frame: pd.DataFrame, freq: str = MINUTE_FREQ) -> pd.DataFrame:
    """Reindex onto a gap-free minute grid so outages become explicit NaN rows."""
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq=freq)
    return frame[~frame.index.duplicated(keep="first")].reindex(full_index)


def _fill_column(series: pd.Series, ffill_limit: int) -> pd.Series:
    """Forward-fill short NaN runs, time-interpolate longer ones."""
    is_na = series.isna()
    if not is_na.any():
        return series

    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")

    short = is_na & (run_len <= ffill_limit)
    long = is_na & (run_len > ffill_limit)

    filled = series.copy()
    filled[short] = series.ffill()[short]
    filled[long] = series.interpolate(method="time")[long]
    # Residual NaN can only sit at a leading edge with no prior value.
    return filled.bfill()


def fill_gaps(frame: pd.DataFrame, ffill_limit: int = FFILL_LIMIT_MINUTES) -> pd.DataFrame:
    """Apply the length-aware fill strategy to every column."""
    return frame.apply(lambda col: _fill_column(col, ffill_limit))


def resample_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the cleaned minute series to hourly resolution.

    Active and reactive power sum to energy (kWh, kVARh); voltage and
    intensity average; sub-metering channels sum their watt-hour counts.
    """
    grouped = frame.resample(HOURLY_FREQ)
    out = pd.DataFrame(index=grouped.mean().index)

    for col in ENERGY_COLUMNS:
        out[col] = grouped[col].sum() * KW_MINUTE_TO_KWH
    for col in MEAN_COLUMNS:
        out[col] = grouped[col].mean()
    for col in SUB_METERING_COLUMNS:
        out[col] = grouped[col].sum()

    out.index.name = "timestamp"
    return out.astype("float32")


def clean_and_resample(frame: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: regular grid -> gap fill -> hourly aggregation."""
    regular = to_regular_grid(frame)
    filled = fill_gaps(regular)
    return resample_hourly(filled)


def save_hourly(frame: pd.DataFrame, path: Path = HOURLY_PATH) -> Path:
    """Persist the hourly frame as parquet, creating the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path
