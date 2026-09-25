"""Offline datasets for the forecasting lab."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoSeries:
    title: str
    values: pd.Series
    description: str
    unit: str
    teaching_point: str


def load_dataset(name: str) -> DemoSeries:
    if name == "AirPassengers (real)":
        path = Path(__file__).parent / "data" / "airpassengers.csv"
        frame = pd.read_csv(path, parse_dates=["date"])
        values = pd.Series(
            frame["value"].to_numpy(dtype=np.float32),
            index=pd.DatetimeIndex(frame["date"], freq="MS"),
            name="actual",
        )
        return DemoSeries(
            name,
            values,
            "Monthly international airline passenger totals, 1949–1960.",
            "thousand passengers",
            "A short series with growing seasonal swings. Notice how a network trained on early years handles later growth.",
        )

    dates = pd.date_range("2005-01-01", periods=240, freq="MS")
    t = np.arange(len(dates), dtype=np.float32)
    month = t % 12
    rng = np.random.default_rng(2026)

    if name == "Steady seasonal demand":
        level = 220 + 0.33 * t
        season = 42 * np.sin(2 * np.pi * (month - 2) / 12)
        demand = level + season + rng.normal(0, 11, len(t))
        description = "A gently rising product with repeatable yearly seasonality."
        teaching = "Try a 12-month lookback, then shorten it to 3 months. Can the model still infer the yearly cycle?"
    elif name == "Promotions and spikes":
        level = 190 + 0.17 * t
        season = 28 * np.sin(2 * np.pi * (month - 1) / 12)
        planned_like_spikes = np.where((month == 10) | (month == 4), 76, 0)
        irregular_spikes = np.where(np.isin(t.astype(int), [34, 86, 137, 192, 218]), 105, 0)
        demand = level + season + planned_like_spikes + irregular_spikes + rng.normal(0, 13, len(t))
        description = "Recurring promotional peaks plus a few irregular demand shocks."
        teaching = "The model sees only past demand. Without a future promotion calendar, irregular spikes cannot be known in advance."
    elif name == "Level shift":
        level = 170 + 0.12 * t + np.where(t >= 158, 83, 0)
        season = 34 * np.sin(2 * np.pi * (month - 2) / 12)
        demand = level + season + rng.normal(0, 11, len(t))
        description = "A product whose demand jumps to a new level late in its history."
        teaching = "Compare validation with test performance: a later structural change can break patterns learned earlier."
    else:
        raise ValueError(f"Unknown dataset: {name}")

    values = pd.Series(np.maximum(demand, 1).astype(np.float32), index=dates, name="actual")
    return DemoSeries(name, values, description, "units", teaching)


DATASET_NAMES = [
    "Steady seasonal demand",
    "Promotions and spikes",
    "Level shift",
    "AirPassengers (real)",
]
