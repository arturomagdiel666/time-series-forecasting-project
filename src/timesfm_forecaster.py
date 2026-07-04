"""Zero-shot TimesFM 2.5 forecaster — optional research add-on.

TimesFM is a HEAVY, OPTIONAL dependency (see ``requirements-timesfm.txt``) with a
large downloaded checkpoint. It is imported LAZILY inside the functions below, so
importing this module — or running the test suite / dashboard — never requires
timesfm to be installed. CI and the Railway deploy install only
``requirements.txt`` and never touch this path; the dashboard reads the committed
prediction parquets, not the model.

Protocol (matches the rest of the project): univariate on the raw hourly TARGET,
strict chronological evaluation, context is ACTUAL past values only (no future
leakage), seed 42.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
CONTEXT_LEN = 1024
MAX_HORIZON = 24
SEED = 42


def load_model(context_len: int = CONTEXT_LEN, max_horizon: int = MAX_HORIZON,
               batch_size: int = 512):
    """Load and compile TimesFM 2.5 200M (torch, CPU).

    timesfm/torch are imported here (not at module top) so the module stays
    importable without the optional dependency. ``max_horizon`` is compiled once
    at 24 so the same model serves both the h=1 and h=24 rolling forecasts.
    """
    import timesfm
    import torch

    # Pin every RNG we touch; TimesFM inference is deterministic given inputs, but
    # keep the seed-42 contract consistent with the rest of the project.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=max_horizon,
            normalize_inputs=True,          # per-window normalization, as configured
            per_core_batch_size=batch_size,
        )
    )
    return model


def _context_windows(values: np.ndarray, positions: list[int], horizon: int,
                     context_len: int) -> list[np.ndarray]:
    """Build leakage-safe context windows for each target position.

    For a target at position ``p`` and forecast ``horizon``, the last context
    point is ``p - horizon`` (i.e. ``horizon`` hours before the target), so that
    forecast step ``horizon`` lands exactly on the target. Every value used is a
    real past observation, so there is no future leakage.
    """
    windows = []
    for p in positions:
        end = p - horizon + 1               # exclusive slice end; last point = p - horizon
        windows.append(values[end - context_len:end])
    return windows


def rolling_forecast(series: pd.Series, test_index: pd.Index, horizon: int,
                     context_len: int = CONTEXT_LEN, batch_size: int = 512,
                     model=None) -> pd.Series:
    """Zero-shot rolling forecast aligned to ``test_index`` for a single horizon.

    Returns a Series of point forecasts indexed by ``test_index``. Pass a
    pre-compiled ``model`` to reuse one load across both horizons.
    """
    series = series.sort_index()
    values = series.to_numpy(dtype="float64")
    position = {ts: i for i, ts in enumerate(series.index)}
    positions = [position[ts] for ts in test_index]

    # Guard: the earliest target must have a full context that ends `horizon`
    # hours before it — otherwise the window would run off the start of history.
    need = context_len + horizon - 1
    if min(positions) < need:
        raise ValueError(
            f"Not enough history: need {need} hours before the first target, "
            f"have {min(positions)}."
        )

    if model is None:
        model = load_model(context_len=context_len, max_horizon=max(horizon, MAX_HORIZON),
                           batch_size=batch_size)

    windows = _context_windows(values, positions, horizon, context_len)
    preds = np.empty(len(windows), dtype="float64")
    # Batch the contexts through the model for CPU throughput.
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        point, _ = model.forecast(horizon=horizon, inputs=chunk)
        preds[i:i + len(chunk)] = point[:, horizon - 1]   # step `horizon` -> the target

    return pd.Series(preds, index=test_index, name="y_pred")
