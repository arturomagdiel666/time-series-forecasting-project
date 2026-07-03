"""Unit tests for leakage-safe feature engineering.

All fixtures are synthetic and deterministic (seed 42); no test touches the real
dataset. Lag/rolling horizons reach 336 hours, so the committed minute-level CSV
fixture is too short here and an hourly frame long enough to exercise the full
warm-up is synthesised in-test instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering as fe

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "synthetic_power_data.csv"

CYCLICAL_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
]


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    """Deterministic hourly series spanning Bastille Day, long enough for lag_336."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2007-07-01", periods=700, freq="h", name="timestamp")
    hour = index.hour.to_numpy()
    daily = 1.0 + 0.6 * np.sin(2 * np.pi * (hour - 6) / 24)
    noise = rng.normal(0, 0.1, len(index))
    values = np.clip(daily + noise, 0.05, None).astype("float32")
    return pd.DataFrame({fe.TARGET: values}, index=index)


def test_cyclical_encodings_bounded(hourly_frame: pd.DataFrame) -> None:
    """sin/cos encodings must stay within the unit range."""
    out = fe.add_cyclical_encodings(hourly_frame)
    for col in CYCLICAL_COLUMNS:
        assert out[col].between(-1.0, 1.0).all(), col


def test_lag_equals_shift(hourly_frame: pd.DataFrame) -> None:
    """Each lag_k column must equal the target shifted by k."""
    out = fe.add_lag_features(hourly_frame)
    for lag in fe.LAGS:
        expected = hourly_frame[fe.TARGET].shift(lag)
        pd.testing.assert_series_equal(
            out[f"lag_{lag}"], expected, check_names=False
        )


def test_rolling_features_no_nan_after_warmup(hourly_frame: pd.DataFrame) -> None:
    """No lag/rolling NaN may survive the documented warm-up drop."""
    features = fe.build_features(hourly_frame)
    cleaned, dropped = fe.drop_warmup(features)

    lag_roll = [c for c in cleaned.columns if c.startswith(("lag_", "roll_"))]
    assert cleaned[lag_roll].isna().sum().sum() == 0
    assert dropped == max(fe.LAGS)  # lag_336 defines the warm-up length here.
    assert len(cleaned) == len(hourly_frame) - dropped


def test_rolling_is_causal(hourly_frame: pd.DataFrame) -> None:
    """The rolling mean must exclude the current hour (shift by one)."""
    out = fe.add_rolling_features(hourly_frame, windows=[24])
    target = hourly_frame[fe.TARGET]
    expected = target.shift(1).rolling(24).mean()
    pd.testing.assert_series_equal(
        out["roll_mean_24"], expected, check_names=False
    )


def test_holiday_flag_on_known_dates(hourly_frame: pd.DataFrame) -> None:
    """Bastille Day is flagged; an ordinary weekday is not."""
    out = fe.add_holiday_flag(hourly_frame)
    bastille = out.loc["2007-07-14 12:00:00", "is_holiday"]
    ordinary = out.loc["2007-07-10 12:00:00", "is_holiday"]  # Tuesday, no holiday
    assert bastille == 1
    assert ordinary == 0


def test_pipeline_runs_on_csv_fixture() -> None:
    """Calendar and cyclical features build on the committed synthetic fixture."""
    raw = pd.read_csv(FIXTURE_CSV, sep=";", na_values=["?", ""])
    index = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    frame = raw[["Global_active_power"]].set_index(pd.DatetimeIndex(index))
    out = fe.add_cyclical_encodings(fe.add_calendar_features(frame))
    assert {"hour", "dayofweek", "is_weekend"}.issubset(out.columns)
    for col in CYCLICAL_COLUMNS:
        assert out[col].between(-1.0, 1.0).all()
