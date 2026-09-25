"""Leakage-aware one-step evaluation and recursive monthly forecasts with PyTorch."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Split:
    train_end: int
    val_end: int
    n: int

    @property
    def train_count(self) -> int:
        return self.train_end

    @property
    def val_count(self) -> int:
        return self.val_end - self.train_end

    @property
    def test_count(self) -> int:
        return self.n - self.val_end


@dataclass(frozen=True)
class Scaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def chronological_split(n: int, train_share: float = 0.60, val_share: float = 0.20) -> Split:
    if not 0 < train_share < 1 or not 0 < val_share < 1 or train_share + val_share >= 1:
        raise ValueError("Split shares must be positive and leave a test period.")
    return Split(int(n * train_share), int(n * (train_share + val_share)), n)


def fit_scaler(training_values: np.ndarray) -> Scaler:
    values = np.asarray(training_values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Training data is empty.")
    std = float(np.std(values))
    return Scaler(float(np.mean(values)), std if std > 1e-8 else 1.0)


def make_windows(scaled: np.ndarray, start: int, end: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Targets are in [start, end); each input ends immediately before its target."""
    if lookback < 1 or start < lookback or end > len(scaled) or end <= start:
        raise ValueError("Invalid window range or lookback.")
    x = np.stack([scaled[i - lookback:i] for i in range(start, end)]).astype(np.float32)
    y = np.asarray(scaled[start:end], dtype=np.float32)
    return x[:, :, None], y[:, None]


class RecurrentForecaster(nn.Module):
    def __init__(self, kind: str, hidden_size: int, layers: int = 1):
        super().__init__()
        if kind == "RNN":
            cell = nn.RNN
        elif kind == "LSTM":
            cell = nn.LSTM
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        self.recurrent = cell(1, hidden_size, num_layers=layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.recurrent(x)
        return self.output(states[:, -1, :])


@dataclass
class ModelResult:
    kind: str
    model: RecurrentForecaster
    history: list[dict[str, float]]
    best_epoch: int
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    future_predictions: np.ndarray
    parameter_count: int


def _predict(model: RecurrentForecaster, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(x).to(device)
        return model(tensor).cpu().numpy().ravel()


def recursive_forecast(
    model: RecurrentForecaster,
    all_scaled: np.ndarray,
    lookback: int,
    horizon: int,
    scaler: Scaler,
    device: torch.device,
) -> np.ndarray:
    if len(all_scaled) < lookback or horizon < 1:
        raise ValueError("Insufficient history or invalid forecast horizon.")
    context = list(np.asarray(all_scaled[-lookback:], dtype=np.float32))
    forecasts = []
    for _ in range(horizon):
        window = np.asarray(context[-lookback:], dtype=np.float32).reshape(1, lookback, 1)
        next_scaled = float(_predict(model, window, device)[0])
        forecasts.append(next_scaled)
        context.append(next_scaled)
    return scaler.inverse(np.asarray(forecasts))


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted arrays must have the same nonzero shape.")
    error = actual - predicted
    denominator = np.abs(actual).sum()
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE": float(np.abs(error).sum() / denominator * 100) if denominator > 0 else float("nan"),
    }


def fit_model(
    kind: str,
    values: np.ndarray,
    split: Split,
    scaler: Scaler,
    lookback: int,
    hidden_size: int,
    layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    horizon: int,
    device: torch.device,
    callback: Callable[[str, int, int, float, float], None] | None = None,
) -> ModelResult:
    values = np.asarray(values, dtype=np.float32)
    if split.n != len(values) or split.train_end <= lookback or split.val_count < 1 or split.test_count < 1:
        raise ValueError("The split must leave training windows and nonempty validation and test periods.")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    scaled = scaler.transform(values)
    train_x, train_y = make_windows(scaled, lookback, split.train_end, lookback)
    val_x, val_y = make_windows(scaled, split.train_end, split.val_end, lookback)
    test_x, _ = make_windows(scaled, split.val_end, split.n, lookback)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    model = RecurrentForecaster(kind, hidden_size, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    val_x_tensor = torch.from_numpy(val_x).to(device)
    val_y_tensor = torch.from_numpy(val_y).to(device)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(x_batch)
            total_count += len(x_batch)
        train_loss = total_loss / total_count
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(val_x_tensor), val_y_tensor).item())
        history.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if callback is not None:
            callback(kind, epoch, epochs, train_loss, val_loss)
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    val_predictions = scaler.inverse(_predict(model, val_x, device))
    test_predictions = scaler.inverse(_predict(model, test_x, device))
    future_predictions = recursive_forecast(model, scaled, lookback, horizon, scaler, device)
    return ModelResult(
        kind=kind,
        model=model,
        history=history,
        best_epoch=best_epoch,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        future_predictions=future_predictions,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )
