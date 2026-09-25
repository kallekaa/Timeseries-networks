import numpy as np
import torch

from datasets import DATASET_NAMES, load_dataset
from forecasting import (
    build_backtest,
    chronological_split,
    error_metrics,
    fit_model,
    fit_scaler,
    make_windows,
    recursive_forecast,
    score_backtest,
    select_backtest,
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
        patience=3, device=torch.device("cpu"),
    )
    future = recursive_forecast(result.model, scaler.transform(values), 12, 6, scaler, torch.device("cpu"))
    assert result.test_predictions.shape == (split.test_count,)
    assert future.shape == (6,)
    assert np.isfinite(future).all()
    assert np.isfinite(error_metrics(values[split.val_end:], result.test_predictions)["MAE"])


class LastValuePlus(torch.nn.Module):
    def forward(self, x):
        return x[:, -1, :] + 0.1


def test_backtest_paths_are_aligned_and_have_no_future_input():
    values = np.arange(1, 51, dtype=np.float32)
    split = chronological_split(len(values))
    scaler = fit_scaler(values[:split.train_end])
    model = LastValuePlus()
    device = torch.device("cpu")
    backtest = build_backtest(values, split, {"RNN": model}, scaler, 3, device, max_horizon=6)
    first_origin = split.val_end - 1
    assert backtest.origins[0] == first_origin
    np.testing.assert_array_equal(backtest.actual[0], values[split.val_end:split.val_end + 6])
    np.testing.assert_array_equal(backtest.predictions["Last month"][0], np.full(6, values[first_origin]))
    np.testing.assert_array_equal(backtest.predictions["Seasonal naïve"][0], values[first_origin - 11:first_origin - 5])
    direct = recursive_forecast(model, scaler.transform(values[:first_origin + 1]), 3, 6, scaler, device)
    np.testing.assert_allclose(backtest.predictions["RNN"][0], direct)
    for row, origin in enumerate(backtest.origins):
        steps = min(6, len(values) - origin - 1)
        assert origin + 1 >= split.val_end
        np.testing.assert_array_equal(backtest.actual[row, :steps], values[origin + 1:origin + 1 + steps])
        np.testing.assert_allclose(
            backtest.predictions["RNN"][row, :steps],
            recursive_forecast(model, scaler.transform(values[:origin + 1]), 3, steps, scaler, device),
        )

    changed_future = values.copy()
    changed_future[split.val_end:] += 1000
    changed = build_backtest(changed_future, split, {"RNN": model}, scaler, 3, device, max_horizon=6)
    np.testing.assert_allclose(changed.predictions["RNN"][0], backtest.predictions["RNN"][0])
    assert np.isfinite(backtest.actual[-1, 0])
    assert np.isnan(backtest.actual[-1, 1:]).all()


def test_complete_origin_cohort_and_signed_bias():
    for name in DATASET_NAMES:
        values = load_dataset(name).values.to_numpy()
        split = chronological_split(len(values))
        backtest = build_backtest(values, split, {}, fit_scaler(values[:split.train_end]), 12, torch.device("cpu"))
        selected = select_backtest(backtest, 24)
        assert len(selected.origins) == split.test_count - 24 + 1
        assert selected.origins[0] == split.val_end - 1
        assert selected.origins[-1] + 24 < split.n
        assert selected.actual.shape == (len(selected.origins), 24)
        assert all(matrix.shape == selected.actual.shape for matrix in selected.predictions.values())
        rows = score_backtest(selected)
        assert len(rows) == 2 * 24
        assert {row["Origins"] for row in rows} == {len(selected.origins)}
    assert len(select_backtest(backtest, 24).origins) == 6  # AirPassengers has 29 test months.

    scores = error_metrics(np.array([10, 20]), np.array([12, 22]))
    assert scores["MAE"] == 2
    assert np.isclose(scores["WAPE"], 100 * 4 / 30)
    assert np.isclose(scores["Bias"], 100 * 4 / 30)
    assert error_metrics(np.array([10, 20]), np.array([8, 18]))["Bias"] < 0
