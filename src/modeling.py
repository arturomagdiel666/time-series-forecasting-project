"""Model matrix construction, naive baselines and SARIMA fitting.

The feature matrix is restricted to information available at forecast time:
calendar/cyclical fields, the holiday flag and past-only lags/rolling stats of
the target. Contemporaneous meter channels and the ``year`` column are excluded
on purpose (see below).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"

TARGET = "Global_active_power"
HOURLY_FREQ = "h"
SEASONAL_PERIOD = 24

# Meter channels are unknown at forecast time and Global_intensity is a near
# linear proxy of the target, so feeding any of them would leak the answer.
LEAKING_CHANNELS = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
# Train years (2006-2009) are disjoint from the test year (2010); a tree cannot
# extrapolate an unseen year, so it is kept for EDA only.
NON_MODEL_COLUMNS = LEAKING_CHANNELS + ["year", "split", TARGET]


@dataclass
class ModelMatrix:
    """Train/test design matrices and aligned targets."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def features(self) -> list[str]:
        return list(self.X_train.columns)


def build_model_matrix(df: pd.DataFrame, verbose: bool = True) -> ModelMatrix:
    """Split by the ``split`` column and drop leaking/non-model columns."""
    feature_cols = [c for c in df.columns if c not in NON_MODEL_COLUMNS]

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    matrix = ModelMatrix(
        X_train=train[feature_cols],
        y_train=train[TARGET],
        X_test=test[feature_cols],
        y_test=test[TARGET],
    )

    if verbose:
        print("model-matrix features:", feature_cols)
        print("X_train:", matrix.X_train.shape, "| X_test:", matrix.X_test.shape)

    return matrix


def naive_forecast(matrix: ModelMatrix, lag: int) -> pd.Series:
    """Seasonal naive prediction: the target value ``lag`` hours earlier.

    The lag columns already hold past target values, so the corresponding
    column on the test rows is exactly the seasonal naive forecast.
    """
    return matrix.X_test[f"lag_{lag}"]


@dataclass
class SarimaFit:
    """A fitted SARIMA result plus the metadata worth reporting."""

    results: object
    train_endog: pd.Series
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    train_window: int
    fit_seconds: float


def fit_sarima(
    y_train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    train_window: int | None = None,
    maxiter: int = 50,
) -> SarimaFit:
    """Fit SARIMAX on the (optionally windowed) recent training tail.

    A full s=24 fit on ~26k points is impractical, so fitting the most recent
    ``train_window`` hours keeps the state-space estimation tractable while
    staying contiguous with the test period for one-step-ahead forecasting.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    endog = y_train.astype("float64")
    if train_window is not None:
        endog = endog.iloc[-train_window:]
    endog = endog.asfreq(HOURLY_FREQ)

    model = SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    start = time.perf_counter()
    results = model.fit(disp=False, maxiter=maxiter, method="lbfgs")
    elapsed = time.perf_counter() - start

    return SarimaFit(
        results=results,
        train_endog=endog,
        order=order,
        seasonal_order=seasonal_order,
        train_window=len(endog),
        fit_seconds=elapsed,
    )


def sarima_one_step(fit: SarimaFit, y_test: pd.Series) -> pd.Series:
    """One-step-ahead forecasts over the test set without re-estimating params.

    The fitted parameters are applied to ``[train tail + test]`` via the Kalman
    filter; in-sample prediction over the test range then conditions each point
    only on genuinely past values.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    endog_test = y_test.astype("float64").asfreq(HOURLY_FREQ)
    full = pd.concat([fit.train_endog, endog_test]).asfreq(HOURLY_FREQ)

    model = SARIMAX(
        full,
        order=fit.order,
        seasonal_order=fit.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    filtered = model.filter(fit.results.params)
    preds = filtered.predict(start=endog_test.index[0], end=endog_test.index[-1])
    return preds.reindex(y_test.index)


@dataclass
class XGBoostFit:
    """A fitted XGBoost model with its predictions and importances."""

    model: object
    y_pred: pd.Series
    best_iteration: int
    importances: pd.Series


def fit_xgboost(
    matrix: ModelMatrix,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> XGBoostFit:
    """Fit a gradient-boosted tree on the raw feature matrix.

    Trees split on raw thresholds, so no scaling is applied. Early stopping uses
    the most recent slice of training as a chronological validation tail, never
    a random fold, to respect the time order.
    """
    from xgboost import XGBRegressor

    cut = int(len(matrix.X_train) * (1 - val_fraction))
    x_tr, y_tr = matrix.X_train.iloc[:cut], matrix.y_train.iloc[:cut]
    x_val, y_val = matrix.X_train.iloc[cut:], matrix.y_train.iloc[cut:]

    # Conservative depth and learning rate with subsampling to limit overfitting
    # on a first, untuned fit; 500 trees give early stopping room to work.
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mae",
        early_stopping_rounds=30,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

    y_pred = pd.Series(model.predict(matrix.X_test), index=matrix.y_test.index)
    importances = pd.Series(
        model.feature_importances_, index=matrix.features
    ).sort_values(ascending=False)

    return XGBoostFit(
        model=model,
        y_pred=y_pred,
        best_iteration=int(model.best_iteration),
        importances=importances,
    )


def save_predictions(
    model_name: str, y_true: pd.Series, y_pred: pd.Series
) -> Path:
    """Persist aligned test predictions for later cross-model aggregation."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"y_true": y_true, "y_pred": pd.Series(y_pred, index=y_true.index)})
    path = PREDICTIONS_DIR / f"{model_name}.parquet"
    frame.to_parquet(path)
    return path
