# Stock Close Predictor

Predict the next close (or next few closes) of any publicly listed stock and visualize it on an animated chart that turns **green** if the model expects a higher close, **red** if lower.

## Two engines, your choice

The sidebar lets you switch between:

- **Classique** (`predict_classic.py`) — the original pipeline: a single `RandomForestRegressor` on a curated set of technical features. Simple, fast, easy to reason about.
- **Avancé** (`predict.py`) — the upgraded pipeline:
  - Ensemble of `RandomForestRegressor` + `LightGBMRegressor` + a naive zero-return baseline (averaged).
  - Direction classifier (`RandomForestClassifier`) producing `P(up)` for the next close.
  - Macro features pulled from `^GSPC` / `^VIX` / `^TNX`.
  - Intraday features (gap, high-low range, close-open range) and cyclical seasonality (day-of-week, month).
  - **Walk-forward** validation on the train tail used to derive **conformal-style** prediction intervals (instead of the naive 1.96 × tree std).
  - **Multi-day recursive forecast** (1 to 10 business days) with intervals that widen with horizon.

## App tabs

- **Predict** — hero number, animated line chart **or** candlestick toggle, model metrics (MAE / RMSE / R² / direction accuracy), feature importances.
- **Backtest** — equity curve of a long/flat strategy that goes long when the model predicts an up move, vs. buy-and-hold over the test window. Hit rate and days-long count.
- **Compare** — pick several tickers, see predicted next-day moves side-by-side in a sortable table and a colored bar chart.

## Stack

- **Data**: [`yfinance`](https://github.com/ranaroussi/yfinance) (zero-config, keyless). Alpaca works too — replace `fetch_history` in either engine.
- **Models**: scikit-learn + LightGBM.
- **UI**: Streamlit + Plotly (frame-based animations).
- **Caching**: Streamlit `@st.cache_data` in memory + joblib on disk under `.prediction_cache/`.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser:
1. In the sidebar pick the engine (Classique / Avancé), the history window, the forecast horizon, and the chart style.
2. In the Predict tab pick a ticker (or type one) and hit **Predict**.
3. Switch to **Backtest** or **Compare** for the extra views.

## Running the tests

```bash
pytest -v
```

The tests run on synthetic OHLCV (no network) and cover both engines plus the walk-forward splitter.

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. On https://share.streamlit.io, click **New app**, point it at this repo, branch, and `app.py`.
3. Streamlit Cloud will install `requirements.txt` automatically. No secrets needed (yfinance is keyless).
4. First load builds the joblib disk cache; subsequent loads for the same `(engine, ticker, period, horizon, macro)` tuple are instant for 30 minutes.

## Files

- `predict.py` — advanced ensemble engine with macro, classifier, conformal CI, multi-day recursive forecast.
- `predict_classic.py` — original single-RF baseline.
- `app.py` — Streamlit UI with three tabs.
- `tests/test_features.py` — pytest smoke tests on synthetic data.
- `requirements.txt` — pinned-floor dependencies.

## Disclaimer

Daily-return prediction on equities is genuinely hard. Out-of-sample R² on real data is typically near zero. Treat the chart and `P(up)` as a directional hint and a way to visualize uncertainty — not as a trading signal.
