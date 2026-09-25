# Demand Forecasting Lab

An interactive Streamlit demo for learning how a simple RNN and an LSTM forecast monthly demand from past actuals. It includes three reproducible synthetic demand patterns and the real **AirPassengers** series. The app runs entirely on your machine after installation.

## Start on Windows PowerShell

```powershell
cd "C:\CODING PROJECTS\Timeseries-networks"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`, and may open it automatically.

### NVIDIA GPU

The app automatically uses CUDA when your installed PyTorch build supports it. For these small monthly datasets a CPU is usually fast enough; a GPU may not be noticeably faster. If the sidebar says CUDA is unavailable and you want to use the GPU, install the Windows/Pip/CUDA build selected by the [official PyTorch installer](https://pytorch.org/get-started/locally/) into the same `.venv`, then restart Streamlit. You can check the environment with:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## How to use it

1. Select a dataset and inspect its train, validation, and test periods.
2. Move the input-window slider to see exactly what the network gets as input.
3. Select **Compare both** and train with a 12-month lookback.
4. In **Test the models**, compare recursive forecasts with the last-month and seasonal naïve baselines at every lead time. Select an origin to see what was known then and what happened afterward.
5. Inspect the one-step and training diagnostics, then compare the backtest with the future forecast.
6. Change the forecast horizon freely. Changing network or training settings requires retraining; changing only the horizon does not.

The split is fixed at 60% training, 20% validation, and 20% test, in chronological order. Scaling uses training observations only. Validation chooses the best epoch via early stopping; test data stays untouched until evaluation. In the main backtest, each forecast origin is the last known month before a recursive forecast. Later test actuals become available only at later origins. Model weights stay fixed throughout the test period. The one-step comparison remains available as a diagnostic.

For a selected horizon, scores use only origins with all forecast months observed. All methods and lead times use that same set of origins. Longer horizons therefore have fewer complete origins; for AirPassengers, a 24-month horizon has only six. The chart and table show MAE, RMSE, WAPE, and signed bias (positive means overforecasting). Forecast-horizon changes select from paths already computed after training.

The model is **univariate**. It does not know future promotions, prices, holidays, or stockouts. The synthetic datasets are generated from fixed random seeds. AirPassengers is bundled for offline use from the [R `datasets` package](https://rweb.stat.umn.edu/R/library/datasets/html/AirPassengers.html); the values were checked against the [Rdatasets CSV](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv). The series is monthly international airline passenger totals, in thousands, from 1949 to 1960.

## Files

- `app.py`: Streamlit user interface and charts
- `forecasting.py`: windowing, scaling, training, rolling-origin evaluation, and recursive forecasts
- `datasets.py` and `data/airpassengers.csv`: example data
- `tests/`: focused checks for data integrity, leakage-sensitive logic, and app interaction

Run tests with `python -m pytest -q` after installing dependencies and pytest.
