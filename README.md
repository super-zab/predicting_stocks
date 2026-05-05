# Stock Close Predictor

A small ML app that predicts the next-day **close** of any publicly listed stock and shows it on an animated chart. The chart turns **green** if the model thinks the stock will close higher, **red** if lower.

## Stack

- **Data**: [`yfinance`](https://github.com/ranaroussi/yfinance) — keyless, works for any ticker on Yahoo Finance. (Alpaca is fine too, but it requires API keys; `yfinance` keeps the setup zero-config. The model code in `predict.py` is data-source-agnostic — swap `fetch_history` if you prefer Alpaca.)
- **Model**: `RandomForestRegressor` on engineered technical features
  (returns, rolling volatility, MA ratios at 3/5/10/20, RSI(14), MACD, Bollinger %B, volume ratio, lagged returns).
- **UI**: Streamlit + Plotly with a frame-based animation.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser:

1. Pick a company from the dropdown (or `Other...` for any ticker).
2. Pick a history window.
3. Hit **Predict**.

The hero number is tomorrow's predicted close. The chart replays the model's out-of-sample predictions vs. the actual price, then ends on tomorrow's forecast with a 95% confidence ribbon (built from the spread of the forest's individual trees).

## Files

- `predict.py` — feature engineering, training, prediction, metrics.
- `app.py` — Streamlit UI and Plotly animation.

## Disclaimer

This is a toy model. Daily-return prediction on equities is extremely noisy and R² on test data is typically near zero — useful for showing uncertainty, not for trading.
