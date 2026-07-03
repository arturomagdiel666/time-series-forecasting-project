# Household Electricity Consumption Forecasting

Short-horizon forecasting of household electricity demand from the UCI
Individual Household Electric Power Consumption dataset. The project compares
three modelling paradigms — classical seasonal ARIMA, gradient-boosted trees
with lag features, and an LSTM — under a strict chronological evaluation
protocol, and ships the result as an interactive dashboard.

## Objectives

1. Forecast next-hour and next-day `Global_active_power` with quantified error.
2. Characterise the dominant temporal patterns (hourly, daily, weekly, seasonal).
3. Determine which paradigm offers the best performance-to-complexity trade-off.

## Data

- Source: UCI Individual Household Electric Power Consumption (CC BY 4.0).
- 2,075,259 minute-level measurements, December 2006 to November 2010, Sceaux, France.
- Roughly 1.25% missing values; target variable `Global_active_power`.

The raw file is not committed. Provide it manually at
`data/raw/household_power_consumption.txt`, or run the loader, which falls back
to a Kaggle download and verifies the file against a known SHA-256 hash.

```bash
python scripts/download_data.py
```

## Method

- **Resampling.** Minute samples are aggregated to hourly resolution; active
  and reactive power convert from kW-minute to kWh and kVARh.
- **Missing values.** Gaps up to 60 minutes are forward-filled; longer gaps use
  time interpolation.
- **Split.** Chronological. Train December 2006 to December 2009, test January
  to November 2010. Cross-validation uses an expanding-window `TimeSeriesSplit`.
- **Metrics.** MAE is primary; RMSE, MAPE, sMAPE and MASE (against a lag-24
  seasonal naive baseline) are reported alongside.
- **Models.** Naive baselines (lag-1, lag-24, lag-168), SARIMA/SARIMAX,
  XGBoost with lag features, and an LSTM in PyTorch.

## Hypotheses

- **H1.** Weekly seasonality is significant (Kruskal-Wallis on day of week).
- **H2.** The series is non-stationary raw and stationary after seasonal differencing.
- **H3.** A tree-based model with lag features can match or beat the LSTM on this
  univariate series.

## Project layout

```
src/                 loading, quality, cleaning, features, EDA, statistical tests
scripts/             data acquisition entry point
notebooks/           executed exploratory analysis
data/processed/      hourly and feature parquet files
reports/figures/     generated EDA figures
tests/               pytest suite with synthetic fixtures
```

## Setup

```bash
python -m pip install -r requirements.txt        # runtime
python -m pip install -r dev-requirements.txt    # development and notebooks
```

Reproducibility: seed 42 throughout; experiments tracked with local MLflow.

## License

MIT. Copyright (c) 2026 Arturo Magdiel.
