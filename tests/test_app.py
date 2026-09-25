from pathlib import Path

from streamlit.testing.v1 import AppTest

from forecasting import select_backtest


def test_training_both_models_and_changing_horizon_keeps_fitted_weights():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception
    before = app.session_state["experiment"]
    assert set(before["results"]) == {"RNN", "LSTM"}
    assert len(select_backtest(before["backtest"], 12).origins) == 37

    horizon = next(widget for widget in app.slider if widget.label == "Forecast horizon (months)")
    horizon.set_value(6).run()
    assert not app.exception
    after = app.session_state["experiment"]
    assert before["backtest"] is after["backtest"]
    assert before["results"]["RNN"].model is after["results"]["RNN"].model
    assert before["results"]["LSTM"].model is after["results"]["LSTM"].model
    assert len(select_backtest(after["backtest"], 6).origins) == 43
