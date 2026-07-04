"""LSTM forecaster for the hourly target (PyTorch, CPU).

The network sees only the scaled target history over a 168-hour lookback, so it
stays within the agreed feature rules. The scaler is fit on the training target
alone; test windows may reach back into the training tail, which is causal (past
values) and therefore not leakage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

LOOKBACK = 168
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
SEED = 42


class LSTMForecaster(nn.Module):
    """Two-layer LSTM mapping a 168-step target window to the next-hour value."""

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])  # last step's hidden state -> scalar


@dataclass
class LSTMResult:
    """LSTM predictions on the test set plus training diagnostics."""

    y_pred: pd.Series
    scaler: StandardScaler
    epochs_run: int
    train_seconds: float
    val_loss_curve: list[float] = field(default_factory=list)


def _make_windows(
    values: np.ndarray, lookback: int, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build (window, next-value) pairs for target positions in [start, stop)."""
    x, y = [], []
    for i in range(max(start, lookback), stop):
        x.append(values[i - lookback : i])
        y.append(values[i])
    x_arr = np.asarray(x, dtype="float32").reshape(-1, lookback, 1)
    y_arr = np.asarray(y, dtype="float32").reshape(-1, 1)
    return x_arr, y_arr


def train_lstm(
    y_train: pd.Series,
    y_test: pd.Series,
    lookback: int = LOOKBACK,
    val_fraction: float = 0.1,
    max_epochs: int = 15,
    patience: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = SEED,
    verbose: bool = False,
) -> LSTMResult:
    """Fit the LSTM on scaled train windows and forecast the test set in kWh."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))
    full = pd.concat([y_train, y_test])
    scaled = scaler.transform(full.to_numpy().reshape(-1, 1)).astype("float32").flatten()

    n_train = len(y_train)
    x_train, y_train_w = _make_windows(scaled, lookback, lookback, n_train)
    x_test, _ = _make_windows(scaled, lookback, n_train, len(full))

    # Chronological validation tail of the training windows (never a random split).
    cut = int(len(x_train) * (1 - val_fraction))
    x_core = torch.from_numpy(x_train[:cut])
    y_core = torch.from_numpy(y_train_w[:cut])
    x_val = torch.from_numpy(x_train[cut:])
    y_val = torch.from_numpy(y_train_w[cut:])

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = model.state_dict()
    epochs_no_improve = 0
    curve: list[float] = []

    start_time = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(x_core))
        for begin in range(0, len(order), batch_size):
            idx = order[begin : begin + batch_size]
            optimizer.zero_grad()
            pred = model(x_core[idx])
            loss = loss_fn(pred, y_core[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        curve.append(val_loss)

        if verbose:
            print(f"  epoch {epoch + 1:>2} val_mse={val_loss:.5f}", flush=True)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        scaled_pred = model(torch.from_numpy(x_test)).numpy()
    y_pred = scaler.inverse_transform(scaled_pred).flatten()

    return LSTMResult(
        y_pred=pd.Series(y_pred, index=y_test.index),
        scaler=scaler,
        epochs_run=len(curve),
        train_seconds=train_seconds,
        val_loss_curve=curve,
    )


def save_scaler(scaler: StandardScaler, name: str = "lstm_scaler.joblib") -> Path:
    """Persist the fitted scaler so inference can reproduce the transform."""
    import joblib

    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / name
    joblib.dump(scaler, path)
    return path
