"""Unit tests for cleaning and hourly resampling.

Gap-fill behaviour is checked on small controlled synthetic series; the
end-to-end resample is checked on the committed synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import data_cleaning as dc
from src import data_loader as dl

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "synthetic_power_data.csv"


def _minute_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2007-01-01", periods=len(values), freq="1min")
    return pd.Series(values, index=index, dtype="float64")


def test_short_gap_is_forward_filled() -> None:
    """A NaN run within the limit carries the last observation forward."""
    series = _minute_series([1.0, 2.0, np.nan, np.nan, 5.0])
    filled = dc._fill_column(series, ffill_limit=60)
    assert filled.iloc[2] == 2.0 and filled.iloc[3] == 2.0


def test_long_gap_is_time_interpolated() -> None:
    """A NaN run beyond the limit is linearly interpolated, not forward-filled."""
    series = _minute_series([1.0, np.nan, np.nan, np.nan, 5.0])
    filled = dc._fill_column(series, ffill_limit=2)
    assert filled.iloc[2] == 3.0  # linear midpoint between 1 and 5
    assert filled.iloc[1] != 1.0  # would equal 1.0 if forward-filled


def test_resample_hourly_rows_and_units() -> None:
    """Two hours of constant power resample to two rows in kWh."""
    index = pd.date_range("2007-01-01", periods=120, freq="1min")
    frame = pd.DataFrame(
        {
            "Global_active_power": 6.0,
            "Global_reactive_power": 0.5,
            "Voltage": 240.0,
            "Global_intensity": 25.0,
            "Sub_metering_1": 1.0,
            "Sub_metering_2": 2.0,
            "Sub_metering_3": 3.0,
        },
        index=index,
    )
    hourly = dc.resample_hourly(frame)
    assert len(hourly) == 2
    # 60 minutes of 6 kW -> 6 kWh; voltage averages; sub-metering sums.
    assert np.isclose(hourly["Global_active_power"].iloc[0], 6.0)
    assert np.isclose(hourly["Voltage"].iloc[0], 240.0)
    assert np.isclose(hourly["Sub_metering_1"].iloc[0], 60.0)


def test_no_residual_nan_after_cleaning() -> None:
    """Cleaning the fixture leaves no missing values in the hourly frame."""
    raw = dl._parse_raw(FIXTURE_CSV)
    hourly = dc.clean_and_resample(raw)
    assert int(hourly.isna().sum().sum()) == 0
