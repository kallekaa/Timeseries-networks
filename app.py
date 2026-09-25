"""Interactive demand forecasting lab: simple RNN versus LSTM."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from datasets import DATASET_NAMES, DemoSeries, load_dataset
from forecasting import chronological_split, error_metrics, fit_model, fit_scaler


BLUE = "#2878B5"
TEAL = "#13A89E"
PURPLE = "#7953B7"
ORANGE = "#E58A35"
GRAY = "#64748B"


def chart_style(fig: go.Figure, title: str, unit: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=height,
        margin=dict(l=20, r=15, t=55, b=25),
        xaxis_title=None,
        yaxis_title=unit,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.11, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(100,116,139,0.18)")
    return fig


def dataset_chart(series: DemoSeries, split) -> go.Figure:
    dates, values = series.values.index, series.values.to_numpy()
    fig = go.Figure()
    fig.add_vrect(x0=dates[0], x1=dates[split.train_end], fillcolor=TEAL, opacity=0.08, line_width=0)
    fig.add_vrect(x0=dates[split.train_end], x1=dates[split.val_end], fillcolor=ORANGE, opacity=0.10, line_width=0)
    fig.add_vrect(x0=dates[split.val_end], x1=dates[-1], fillcolor=PURPLE, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name="Actual", line=dict(color=BLUE, width=2.5)))
    fig.add_vline(x=dates[split.train_end], line_color=ORANGE, line_dash="dash")
    fig.add_vline(x=dates[split.val_end], line_color=PURPLE, line_dash="dash")
    return chart_style(fig, "Historical actuals and chronological split", series.unit)


def window_chart(series: DemoSeries, target: int, lookback: int) -> go.Figure:
    dates, values = series.values.index, series.values.to_numpy()
    context_start = max(0, target - lookback - 12)
    context_end = min(len(values), target + 7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates[context_start:context_end], y=values[context_start:context_end],
        mode="lines+markers", name="Actuals", line=dict(color=GRAY, width=1.8), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=dates[target - lookback:target], y=values[target - lookback:target],
        mode="lines+markers", name="Input window", line=dict(color=TEAL, width=4), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=[dates[target]], y=[values[target]], mode="markers", name="Target to predict",
        marker=dict(color=ORANGE, size=15, symbol="diamond"),
    ))
    return chart_style(fig, f"One training example: {lookback} months in → next month out", series.unit, 360)


def evaluation_chart(series: DemoSeries, split, results: dict, baseline: np.ndarray) -> go.Figure:
    dates, values = series.values.index, series.values.to_numpy()
    start = max(split.val_end - 12, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[start:], y=values[start:], name="Actual", line=dict(color=GRAY, width=3)))
    test_dates = dates[split.val_end:]
    fig.add_trace(go.Scatter(x=test_dates, y=baseline, name="Seasonal naïve", line=dict(color=ORANGE, width=2, dash="dot")))
    for kind, result in results.items():
        color = BLUE if kind == "RNN" else PURPLE
        fig.add_trace(go.Scatter(x=test_dates, y=result.test_predictions, name=kind, line=dict(color=color, width=2.4)))
    fig.add_vline(x=dates[split.val_end], line_color=PURPLE, line_dash="dash")
    return chart_style(fig, "Held-out test: rolling one-step predictions", series.unit)


def forecast_chart(series: DemoSeries, horizon: int, results: dict, baseline: np.ndarray) -> go.Figure:
    dates, values = series.values.index, series.values.to_numpy()
    future_dates = pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[-36:], y=values[-36:], name="Observed", line=dict(color=GRAY, width=3)))
    fig.add_trace(go.Scatter(x=future_dates, y=baseline, name="Seasonal naïve", line=dict(color=ORANGE, width=2, dash="dot")))
    for kind, result in results.items():
        color = BLUE if kind == "RNN" else PURPLE
        fig.add_trace(go.Scatter(
            x=np.r_[dates[-1:], future_dates], y=np.r_[values[-1:], result.future_predictions],
            name=kind, line=dict(color=color, width=2.7),
        ))
    fig.add_vline(x=future_dates[0], line_color=PURPLE, line_dash="dash")
    return chart_style(fig, "Future forecast: recursive multi-step predictions", series.unit)


def loss_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    for kind, result in results.items():
        color = BLUE if kind == "RNN" else PURPLE
        hist = pd.DataFrame(result.history)
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["train"], name=f"{kind} train", line=dict(color=color, dash="dot")))
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["validation"], name=f"{kind} validation", line=dict(color=color, width=2.5)))
    return chart_style(fig, "Learning curves · standardized mean squared error", "MSE", 360)


def _baseline_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    last_year = values[-12:]
    return np.resize(last_year, horizon)


def _metrics_frame(series: DemoSeries, split, results: dict) -> pd.DataFrame:
    values = series.values.to_numpy()
    val_actual = values[split.train_end:split.val_end]
    test_actual = values[split.val_end:]
    rows = []
    for name, val_pred, test_pred in [
        ("Last month", values[split.train_end - 1:split.val_end - 1], values[split.val_end - 1:-1]),
        ("Seasonal naïve (12-month lag)", values[split.train_end - 12:split.val_end - 12], values[split.val_end - 12:-12]),
        *[(kind, result.val_predictions, result.test_predictions) for kind, result in results.items()],
    ]:
        val_metrics = error_metrics(val_actual, val_pred)
        test_metrics = error_metrics(test_actual, test_pred)
        rows.append({
            "Method": name,
            "Validation MAE": round(val_metrics["MAE"], 2),
            "Test MAE": round(test_metrics["MAE"], 2),
            "Test RMSE": round(test_metrics["RMSE"], 2),
            "Test WAPE %": round(test_metrics["WAPE"], 2),
        })
    return pd.DataFrame(rows).set_index("Method")


def main() -> None:
    st.set_page_config(page_title="Demand Forecasting Lab", page_icon="📈", layout="wide")
    st.markdown("""
    <style>
      .block-container {padding-top: 2.1rem; max-width: 1450px;}
      h1, h2, h3 {letter-spacing: -0.025em;}
      div[data-testid="stMetric"] {border: 1px solid rgba(100,116,139,.22); border-radius: 12px; padding: 12px 16px;}
      .eyebrow {color: #13A89E; font-size: .82rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase;}
      .hero-sub {color: #64748B; font-size: 1.08rem; margin-top: -.6rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Interactive learning demo</div>', unsafe_allow_html=True)
    st.title("Demand forecasting with RNNs and LSTMs")
    st.markdown('<div class="hero-sub">Choose a demand pattern, train a model, and compare what it predicts with what actually happened.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("1 · Data")
        dataset_name = st.selectbox("Example series", DATASET_NAMES)
        series = load_dataset(dataset_name)
        st.caption(series.description)
        st.header("2 · Experiment")
        model_choice = st.selectbox("Network", ["Compare both", "RNN", "LSTM"])
        lookback = st.slider("Past months shown to model", 3, 36, 12, help="Number of past actuals in each input window.")
        horizon = st.slider("Future months to forecast", 1, 24, 12)
        st.header("3 · Training")
        hidden_size = st.select_slider("Hidden units", options=[8, 16, 32, 64], value=32, help="Size of each recurrent hidden state. Larger models can capture more patterns but may overfit.")
        layers = st.select_slider("Recurrent layers", options=[1, 2, 3], value=1, help="Number of recurrent layers stacked on top of one another.")
        epochs = st.slider("Maximum epochs", 20, 200, 80, step=10, help="An epoch is one pass through all training windows. Early stopping may finish sooner.")
        learning_rate = st.select_slider("Learning rate", options=[0.0003, 0.001, 0.003, 0.01], value=0.003, format_func=lambda x: f"{x:g}", help="Step size for the Adam optimizer. Too large can make training unstable; too small can train slowly.")
        with st.expander("More training controls"):
            batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32, help="Number of windows used per weight update.")
            patience = st.slider("Early stopping patience", 5, 30, 12, help="Stop after this many epochs without a new best validation loss.")
        has_cuda = torch.cuda.is_available()
        device_options = ["Auto", "CPU"] + (["NVIDIA GPU"] if has_cuda else [])
        device_choice = st.selectbox("Compute device", device_options)
        device = torch.device("cuda" if has_cuda and device_choice != "CPU" else "cpu")
        if has_cuda:
            st.caption(f"Available: {torch.cuda.get_device_name(0)} · using {device.type.upper()}")
        else:
            st.caption("CUDA is unavailable in this PyTorch installation; training uses CPU.")
        train_clicked = st.button("Train and evaluate", type="primary", width="stretch")

    values = series.values.to_numpy(dtype=np.float32)
    split = chronological_split(len(values))
    st.caption(
        f"Monthly data · {len(values)} observations · "
        f"train {split.train_count} / validation {split.val_count} / test {split.test_count} months"
    )
    if split.train_end <= lookback + 5:
        st.error("The lookback is too long for this dataset and training split.")
        st.stop()

    signature = (dataset_name, model_choice, lookback, horizon, hidden_size, layers, epochs, learning_rate, batch_size, patience, device.type)
    if train_clicked:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        scaler = fit_scaler(values[:split.train_end])
        kinds = ["RNN", "LSTM"] if model_choice == "Compare both" else [model_choice]
        progress = st.progress(0, text="Preparing training windows…")
        status = st.empty()
        results = {}
        for index, kind in enumerate(kinds):
            def update(label: str, epoch: int, max_epochs: int, train_loss: float, val_loss: float) -> None:
                fraction = (index + epoch / max_epochs) / len(kinds)
                progress.progress(min(fraction, 1.0), text=f"Training {label} · epoch {epoch}/{max_epochs}")
                if epoch == 1 or epoch % 5 == 0:
                    status.caption(f"{label}: train loss {train_loss:.4f} · validation loss {val_loss:.4f}")

            results[kind] = fit_model(
                kind, values, split, scaler, lookback, hidden_size, layers, epochs,
                batch_size, learning_rate, patience, horizon, device, update,
            )
        progress.empty()
        status.empty()
        st.session_state["experiment"] = {"signature": signature, "results": results}

    saved = st.session_state.get("experiment")
    current = saved is not None and saved["signature"] == signature
    if saved is not None and not current:
        st.info("Settings changed. Select **Train and evaluate** to update the results below.")

    explore_tab, score_tab, forecast_tab, learn_tab = st.tabs([
        "Explore the data", "Test the models", "Forecast ahead", "How it works",
    ])

    with explore_tab:
        left, right = st.columns([3, 1], gap="large")
        with left:
            st.plotly_chart(dataset_chart(series, split), width="stretch")
        with right:
            st.subheader("What to notice")
            st.write(series.teaching_point)
            st.markdown("**Green**: fit weights. **Amber**: choose when to stop. **Purple**: final test period.")
            st.caption("The shaded periods are fixed in time. Rows are never randomly split.")
        st.subheader("See the input window")
        example_target = st.slider(
            "Month the model tries to predict", lookback, len(values) - 1,
            min(split.train_end - 1, len(values) - 1),
        )
        st.plotly_chart(window_chart(series, example_target, lookback), width="stretch")
        st.caption(
            f"For {series.values.index[example_target]:%B %Y}, the network receives the previous "
            f"{lookback} actuals and is trained to estimate the orange point. Every month creates another example."
        )

    with score_tab:
        if not current:
            st.info("Choose your settings and select **Train and evaluate** to see the held-out comparison.")
        else:
            results = saved["results"]
            baseline = values[split.val_end - 12:-12]
            metrics = _metrics_frame(series, split, results)
            st.subheader("Performance on unseen test months")
            st.dataframe(metrics, width="stretch")
            st.caption("MAE and RMSE are in the series unit; lower is better. WAPE is total absolute error divided by total actual demand. Compare validation and test MAE to spot changing conditions.")
            st.plotly_chart(evaluation_chart(series, split, results, baseline), width="stretch")
            st.info(
                "These are rolling **one-step** predictions. At each test month, the input window may include "
                "actuals from earlier test months. Model weights never update on validation or test data."
            )
            st.plotly_chart(loss_chart(results), width="stretch")
            cols = st.columns(len(results))
            for col, (kind, result) in zip(cols, results.items()):
                with col:
                    st.metric(f"{kind} best epoch", result.best_epoch)
                    st.caption(f"{result.parameter_count:,} trainable parameters; stopped after {len(result.history)} epochs.")

    with forecast_tab:
        if not current:
            st.info("Train a model to produce a future forecast.")
        else:
            results = saved["results"]
            baseline_future = _baseline_forecast(values, horizon)
            st.plotly_chart(forecast_chart(series, horizon, results, baseline_future), width="stretch")
            st.warning(
                "Future months use each prior prediction as input. Errors can compound, so this curve is a "
                "harder task than the rolling one-step test. No future promotions, prices, holidays, or "
                "business decisions are supplied to the model. These are point forecasts, not uncertainty ranges."
            )
            st.caption("For this learning experiment, network weights were fit only on the first 60% of history. The latest actuals are used as forecast inputs. A production model would normally be refit on all available history after evaluation.")
            future_dates = pd.date_range(series.values.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            forecast_frame = pd.DataFrame({"date": future_dates, "seasonal_naive": baseline_future})
            for kind, result in results.items():
                forecast_frame[kind.lower()] = result.future_predictions
            st.dataframe(forecast_frame.style.format(precision=1), width="stretch", hide_index=True)
            st.download_button(
                "Download forecast CSV", forecast_frame.to_csv(index=False).encode("utf-8"),
                file_name="demand_forecast_demo.csv", mime="text/csv",
            )
            if any(np.any(result.future_predictions < 0) for result in results.values()):
                st.caption("A network produced a negative forecast. Raw outputs are shown so you can see that behavior.")

    with learn_tab:
        st.subheader("RNN and LSTM: same task, different memory")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""
            **Simple RNN**

            At each month, it combines the new demand value with a hidden state from the prior month. The final hidden state feeds a dense layer that predicts the next value. Its compact design is a useful starting point, but information from distant months can fade during training.

            `past demand → recurrent hidden state → next-month demand`
            """)
        with col2:
            st.markdown("""
            **LSTM**

            It also processes the history one month at a time, but uses input, forget, and output gates plus a cell state. Those paths can help preserve relevant information across a longer window. More parameters do not guarantee a better forecast.

            `past demand → gates + cell state → next-month demand`
            """)
        st.divider()
        st.subheader("The experiment, step by step")
        st.markdown("""
        1. **Split by time:** first 60% for fitting, next 20% for validation, final 20% for testing.
        2. **Scale without leakage:** mean and standard deviation come from the training period only.
        3. **Build windows:** each input contains the chosen number of past actuals; the target is the next month.
        4. **Train:** Adam minimizes mean squared error. Gradient clipping helps stabilize updates.
        5. **Stop early:** the best validation epoch is restored. The test period never selects weights or stopping time.
        6. **Evaluate fairly:** compare the networks with last-month and seasonal naïve forecasts on the same test months.
        7. **Forecast ahead:** roll predictions forward one month at a time from the end of observed history.
        """)
        st.subheader("Suggested experiments")
        st.markdown("""
        - On **Steady seasonal demand**, compare 3 versus 12 versus 24 months of history.
        - On **Promotions and spikes**, inspect which peaks remain unpredictable from demand history alone.
        - On **Level shift**, compare validation and test errors around the change in demand level.
        - On **AirPassengers**, see whether the seasonal naïve baseline beats a neural network trained on only a few years.
        - Try a larger hidden state. Watch whether validation loss improves even when training loss keeps falling.
        """)
        st.caption(
            "This is a one-series, univariate learning demo. For production planning, also test multiple forecast origins, "
            "business drivers, intermittent demand handling, forecast bias, uncertainty, and the hierarchy of SKUs and locations."
        )


if __name__ == "__main__":
    main()
