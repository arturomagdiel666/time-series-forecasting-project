# Decision log

Key methodological decisions, each with its rationale and the alternative considered. Referenced from the [README](README.md).

## Data and evaluation

**Chronological train/test split (not random).**
- *Rationale:* a time series must be validated on a window that strictly follows training; a random split places future observations in the training set and leaks the autocorrelation the model exploits.
- *Alternative:* random hold-out — rejected as leakage that inflates scores and misrepresents deployment.

**TimeSeriesSplit expanding-window CV (not KFold).**
- *Rationale:* each validation fold sits after its training slice, the in-sample analogue of the chronological split.
- *Alternative:* KFold / StratifiedKFold — rejected for shuffling time order.

**Hourly resampling.**
- *Rationale:* the operationally relevant granularity; suppresses minute-level sensor noise while preserving daily and weekly structure.
- *Alternative:* modelling at minute resolution — rejected as noisy and ~60x larger for no forecasting benefit at these horizons.

**Imputation: forward-fill gaps ≤60 min, time-interpolate longer gaps.**
- *Rationale:* consumption rarely jumps within an hour, so carrying the last reading forward is defensible for short outages; longer gaps need interpolation to avoid flat-lining a whole period.
- *Alternative:* drop missing rows — rejected because it breaks the regular time grid the lag features depend on.

## Metrics

**MAE primary; MASE with a lag-24 seasonal-naive denominator (0.6551).**
- *Rationale:* MAE is robust and in the target's own kWh units; MASE against the daily seasonal naive makes "does it beat same-hour-yesterday" explicit (MASE < 1 = yes).
- *Alternative:* RMSE or MAPE as primary — kept as secondary; RMSE overweights rare spikes and MAPE is unstable near zero demand.

## Features

**Exclude contemporaneous meter channels (Global_intensity, Global_reactive_power, Voltage, Sub_metering_*).**
- *Rationale:* they are unknown at forecast time, and Global_intensity is a near-linear proxy of the target — including any would leak the answer.
- *Alternative:* use them as regressors — rejected as target leakage.

**Exclude `year` from the model matrix.**
- *Rationale:* train years (2006–2009) are disjoint from the test year (2010); a tree cannot extrapolate an unseen category. Kept for EDA only.
- *Alternative:* keep `year` — rejected as a spurious, non-generalising split variable.

**Causal lags/rolling computed so the test set may read the training tail but never the future.**
- *Rationale:* mirrors real inference (at time t only past values are known); rolling stats are shifted by one so the current hour never enters its own window.
- *Alternative:* centred rolling windows — rejected as forward-looking leakage.

## Models

**SARIMA (1,0,1)(1,1,1,24) on the last ~90 days of training.**
- *Rationale:* orders are informed by ACF/PACF and H2 (D=1 at s=24; d=0 since the raw level is mean-reverting; p=q=P=Q=1 for intraday and daily structure). A full s=24 fit on ~26k points is impractical, so the recent window keeps estimation tractable while staying contiguous with the test period.
- *Alternative:* full-history fit or an auto-order grid search — rejected for cost with no expected accuracy gain over the informed orders.

**LSTM: fixed architecture (2-layer, hidden 64, dropout 0.2, 168h→1h), not tuned.**
- *Rationale:* its role is to test H3 (can a deep sequence model beat trees here), not to win; a fixed, reasonable configuration is enough to answer that.
- *Alternative:* extensive LSTM tuning — rejected as disproportionate effort for a model that already trails XGBoost and SARIMA.

**XGBoost with lag features as the deployed model.**
- *Rationale:* lowest test and CV error at the lowest operational cost (seconds to fit, no scaling, CPU-friendly, interpretable importances).
- *Alternative:* SARIMA or LSTM — rejected on the accuracy-and-cost trade-off.

**Next-day forecast via a direct h=24 model.**
- *Rationale:* a single model predicting y[t+24] from information available at t is simple and leakage-free (calendar of t+24 is known; lags/rolling are taken at the origin).
- *Alternative:* recursive multi-step or a multi-output model — not compared here; noted as a limitation.

## Engineering

**Python pinned to 3.12 (3.12.10).**
- *Rationale:* PyTorch 2.x has no wheels for Python 3.14 (the machine default); 3.12 is required for the LSTM and for reproducible clone-and-run and Railway deployment.
- *Alternative:* Python 3.14 — rejected as incompatible with a fixed dependency.

**Global seed 42.**
- *Rationale:* reproducibility across Python, NumPy and PyTorch.

**MLflow local file store (`mlruns/`).**
- *Rationale:* zero-infrastructure experiment tracking suitable for a portfolio project; every run logs params, all five metrics and prediction artifacts.
- *Alternative:* a database-backed or hosted tracking server — unnecessary overhead at this scale.
