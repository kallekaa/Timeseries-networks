import numpy as np
import torch

from datasets import DATASET_NAMES, load_dataset
from forecasting import (
    chronological_split,
    error_metrics,
    fit_model,
    fit_scaler,
    make_windows,
)


def test_all_datasets_are_monthly_and_positive():
    for name in DATASET_NAMES:
        series = load_dataset(name).values
        assert len(series) >= 140
        assert series.index.is_monotonic_increasing
        assert series.index.freqstr == "MS"
        assert (series.to_numpy() > 0).all()
    air = load_dataset("AirPassengers (real)").values
    assert len(air) == 144
    assert air.iloc[0] == 112
    assert air.iloc[-1] == 432


def test_scaler_and_windows_do_not_use_future_targets():
    raw = np.array([10, 11, 12, 13, 14, 100, 101, 102], dtype=np.float32)
    scaler = fit_scaler(raw[:5])
    assert scaler.mean == 12
    scaled = scaler.transform(raw)
    x, y = make_windows(scaled, start=5, end=8, lookback=3)
    np.testing.assert_array_equal(x[0, :, 0], scaled[2:5])
    np.testing.assert_array_equal(y[:, 0], scaled[5:8])
    assert np.max(x[0]) < np.min(y)


def test_small_end_to_end_training_and_future_forecast():
    values = load_dataset("Steady seasonal demand").values.to_numpy()
    split = chronological_split(len(values))
    scaler = fit_scaler(values[:split.train_end])
    result = fit_model(
        "RNN", values, split, scaler, lookback=12, hidden_size=8,
        layers=1, epochs=3, batch_size=32, learning_rate=0.003,
        patience=3, horizon=6, device=torch.device("cpu"),
    )
    assert result.test_predictions.shape == (split.test_count,)
    assert result.future_predictions.shape == (6,)
    assert np.isfinite(result.future_predictions).all()
    assert np.isfinite(error_metrics(values[split.val_end:], result.test_predictions)["MAE"])
