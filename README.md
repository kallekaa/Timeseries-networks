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
4. Compare MAE, RMSE, and WAPE with the last-month and seasonal naïve baselines.
5. Inspect the learning curves, then compare the one-step test plot with the recursive future forecast.
6. Change one setting at a time and retrain to understand its effect.

The split is fixed at 60% training, 20% validation, and 20% test, in chronological order. Scaling uses training observations only. Validation chooses the best epoch via early stopping; test data stays untouched until evaluation. Test predictions are rolling one-step predictions: earlier actuals in the test period can appear in later input windows. Future predictions are recursive and use earlier predictions instead.

The model is **univariate**. It does not know future promotions, prices, holidays, or stockouts. The synthetic datasets are generated from fixed random seeds. AirPassengers is bundled for offline use from the [R `datasets` package](https://rweb.stat.umn.edu/R/library/datasets/html/AirPassengers.html); the values were checked against the [Rdatasets CSV](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv). The series is monthly international airline passenger totals, in thousands, from 1949 to 1960.

## Files

- `app.py`: Streamlit user interface and charts
- `forecasting.py`: windowing, scaling, training, evaluation, and recursive forecasts
- `datasets.py` and `data/airpassengers.csv`: example data
- `tests/test_forecasting.py`: focused checks for data integrity and leakage-sensitive logic

Run tests with `python -m pytest -q` after installing dependencies and pytest.
