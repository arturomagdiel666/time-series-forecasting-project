"""Statistical tests for stationarity, seasonality and residual structure.

ADF and KPSS are paired deliberately: they carry opposite null hypotheses, so
agreement gives a stronger stationarity verdict than either alone. Seasonality
uses Kruskal-Wallis rather than ANOVA because hourly energy is heavy-tailed and
skewed, violating the normality ANOVA assumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

SEASONAL_PERIOD = 24
LJUNG_BOX_LAGS = [24, 168, 720]
ALPHA = 0.05


@dataclass
class StationarityResult:
    """Paired ADF/KPSS outcome for one series."""

    label: str
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float

    @property
    def adf_stationary(self) -> bool:
        """ADF rejects a unit root -> stationary."""
        return self.adf_pvalue < ALPHA

    @property
    def kpss_stationary(self) -> bool:
        """KPSS fails to reject stationarity -> stationary."""
        return self.kpss_pvalue >= ALPHA

    @property
    def verdict(self) -> str:
        """Combine both tests into a single stationarity label."""
        if self.adf_stationary and self.kpss_stationary:
            return "stationary"
        if not self.adf_stationary and not self.kpss_stationary:
            return "non-stationary"
        return "inconclusive"


def _adf(series: pd.Series) -> tuple[float, float]:
    stat, pvalue, *_ = adfuller(series, autolag="AIC")
    return float(stat), float(pvalue)


def _kpss(series: pd.Series) -> tuple[float, float]:
    # regression="c": test level stationarity around a constant.
    stat, pvalue, *_ = kpss(series, regression="c", nlags="auto")
    return float(stat), float(pvalue)


def stationarity(series: pd.Series, label: str) -> StationarityResult:
    """Run paired ADF and KPSS tests on a series."""
    clean = series.dropna()
    adf_stat, adf_p = _adf(clean)
    kpss_stat, kpss_p = _kpss(clean)
    return StationarityResult(label, adf_stat, adf_p, kpss_stat, kpss_p)


def seasonal_difference(series: pd.Series, period: int = SEASONAL_PERIOD) -> pd.Series:
    """Return the lag-``period`` seasonal difference."""
    return series.diff(period).dropna()


def stationarity_raw_and_differenced(
    series: pd.Series, period: int = SEASONAL_PERIOD
) -> dict[str, StationarityResult]:
    """Test stationarity on the raw series and its seasonal difference."""
    return {
        "raw": stationarity(series, "raw"),
        "seasonal_diff": stationarity(
            seasonal_difference(series, period), f"seasonal_diff_{period}"
        ),
    }


@dataclass
class KruskalResult:
    """Kruskal-Wallis outcome for one grouping factor."""

    factor: str
    statistic: float
    pvalue: float

    @property
    def significant(self) -> bool:
        """Reject equal-distribution null -> the factor matters."""
        return self.pvalue < ALPHA


def kruskal_by(series: pd.Series, factor: str) -> KruskalResult:
    """Kruskal-Wallis test of the series across a calendar factor."""
    idx = series.index
    attribute = {"hour": idx.hour, "day_of_week": idx.dayofweek}[factor]
    frame = pd.DataFrame({"value": series.to_numpy(), "group": attribute})
    groups = [g["value"].to_numpy() for _, g in frame.groupby("group")]
    stat, pvalue = kruskal(*groups)
    return KruskalResult(factor, float(stat), float(pvalue))


def ljung_box(series: pd.Series, lags: list[int] = LJUNG_BOX_LAGS) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation at the requested lags."""
    return acorr_ljungbox(series.dropna(), lags=lags, return_df=True)
