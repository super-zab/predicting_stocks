"""Stock close-price prediction with a Random Forest on engineered features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "return", "volatility",
    "ma_ratio_3", "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "rsi", "macd", "macd_signal", "bb_pct",
    "vol_ratio",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
]


def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return"] = out["close"].pct_change()
    out["volatility"] = out["return"].rolling(5).std()

    for w in (3, 5, 10, 20):
        out[f"ma_{w}"] = out["close"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["close"] / out[f"ma_{w}"] - 1

    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    bb_mean = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std()
    upper = bb_mean + 2 * bb_std
    lower = bb_mean - 2 * bb_std
    out["bb_pct"] = (out["close"] - lower) / (upper - lower)

    vol_ma = out["volume"].rolling(5).mean()
    out["vol_ratio"] = out["volume"] / vol_ma.replace(0, np.nan)

    for lag in (1, 2, 3, 5):
        out[f"return_lag_{lag}"] = out["return"].shift(lag)

    out["target_return"] = out["return"].shift(-1)
    return out.dropna()


def train_and_predict(ticker: str, period: str = "5y") -> dict:
    raw = fetch_history(ticker, period=period)
    df = add_features(raw)

    if len(df) < 100:
        raise ValueError(f"Not enough history for '{ticker}' ({len(df)} usable rows).")

    X = df[FEATURES].values
    y = df["target_return"].values
    close = df["close"].values

    test_size = max(30, int(len(df) * 0.2))
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    close_test = close[-test_size:]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    pred_returns = model.predict(X_test_s)
    tree_preds = np.stack([t.predict(X_test_s) for t in model.estimators_])
    pred_std = tree_preds.std(axis=0)

    pred_prices = close_test * (1 + pred_returns)
    actual_prices = close_test * (1 + y_test)
    lower_prices = close_test * (1 + pred_returns - 1.96 * pred_std)
    upper_prices = close_test * (1 + pred_returns + 1.96 * pred_std)

    mae = float(mean_absolute_error(actual_prices, pred_prices))
    rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
    r2 = float(r2_score(actual_prices, pred_prices))

    last_features = df[FEATURES].iloc[[-1]].values
    last_features = np.clip(last_features, X_train.min(axis=0), X_train.max(axis=0))
    last_scaled = scaler.transform(last_features)

    next_return = float(model.predict(last_scaled)[0])
    next_tree = np.array([t.predict(last_scaled)[0] for t in model.estimators_])
    next_std = float(next_tree.std())
    last_close = float(df["close"].iloc[-1])
    next_price = last_close * (1 + next_return)

    last_date = df.index[-1]
    next_date = last_date + pd.tseries.offsets.BDay(1)

    return {
        "ticker": ticker.upper(),
        "history": raw,
        "test_dates": df.index[-test_size:],
        "actual_prices": actual_prices,
        "pred_prices": pred_prices,
        "lower_prices": lower_prices,
        "upper_prices": upper_prices,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "last_close": last_close,
        "last_date": last_date,
        "next_date": next_date,
        "next_price": next_price,
        "next_lower": last_close * (1 + next_return - 1.96 * next_std),
        "next_upper": last_close * (1 + next_return + 1.96 * next_std),
    }
