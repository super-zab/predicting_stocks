"""Stock close-price prediction.

Pipeline:
  fetch_history -> add_features -> {regression head, classification head}
  Models: Random Forest, LightGBM (optional), naive baseline. Ensemble averages
  the predictive returns of all enabled regression models.
  Validation: walk-forward (expanding window) on the last K folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


PRICE_FEATURES = [
    "return", "volatility",
    "ma_ratio_3", "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "rsi", "macd", "macd_signal", "bb_pct",
    "vol_ratio",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
]

INTRADAY_FEATURES = ["gap_open", "hl_range", "co_range"]
SEASONAL_FEATURES = ["dow_sin", "dow_cos", "month_sin", "month_cos"]
MACRO_FEATURES = ["spx_return", "vix_level", "vix_change", "tnx_change"]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    df = _flatten_columns(df)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_macro(period: str = "5y") -> pd.DataFrame | None:
    try:
        raw = yf.download(
            ["^GSPC", "^VIX", "^TNX"],
            period=period, progress=False, auto_adjust=True,
        )
    except Exception:
        return None
    if raw.empty:
        return None
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(close, pd.Series):
        return None
    out = pd.DataFrame(index=close.index)
    out["spx_return"] = close["^GSPC"].pct_change()
    out["vix_level"] = close["^VIX"]
    out["vix_change"] = close["^VIX"].pct_change()
    out["tnx_change"] = close["^TNX"].pct_change()
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    out["return"] = out["close"].pct_change()
    out["volatility"] = out["return"].rolling(5).std()

    for w in (3, 5, 10, 20):
        ma = out["close"].rolling(w).mean()
        out[f"ma_{w}"] = ma
        out[f"ma_ratio_{w}"] = out["close"] / ma - 1

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
    out["bb_pct"] = (out["close"] - lower) / (upper - lower).replace(0, np.nan)

    vol_ma = out["volume"].rolling(5).mean()
    out["vol_ratio"] = out["volume"] / vol_ma.replace(0, np.nan)

    for lag in (1, 2, 3, 5):
        out[f"return_lag_{lag}"] = out["return"].shift(lag)

    # Intraday
    prev_close = out["close"].shift(1)
    out["gap_open"] = (out["open"] - prev_close) / prev_close
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["co_range"] = (out["close"] - out["open"]) / out["open"]

    # Seasonality (cyclical encoding)
    dow = out.index.dayofweek.to_numpy()
    month = out.index.month.to_numpy() - 1
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Macro (joined on date, forward-filled across small gaps)
    if macro is not None:
        out = out.join(macro, how="left").ffill()

    out["target_return"] = out["return"].shift(-1)
    out["target_up"] = (out["target_return"] > 0).astype(int)
    return out.dropna()


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = list(PRICE_FEATURES) + list(INTRADAY_FEATURES) + list(SEASONAL_FEATURES)
    cols += [c for c in MACRO_FEATURES if c in df.columns]
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    build: Callable
    needs_scaling: bool = True


def make_rf_regressor():
    return RandomForestRegressor(
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )


def make_lgbm_regressor():
    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=10, subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1, verbose=-1,
    )


def make_rf_classifier():
    return RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )


def available_regressors() -> list[ModelSpec]:
    specs = [ModelSpec("random_forest", make_rf_regressor, needs_scaling=True)]
    if HAS_LGBM:
        specs.append(ModelSpec("lightgbm", make_lgbm_regressor, needs_scaling=False))
    return specs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def walk_forward_indices(n: int, n_folds: int = 5, min_train: int = 200) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds over the tail of the series."""
    fold_size = max(20, (n - min_train) // n_folds)
    if fold_size <= 0:
        return
    start = max(min_train, n - fold_size * n_folds)
    for i in range(n_folds):
        train_end = start + i * fold_size
        test_end = min(train_end + fold_size, n)
        if train_end >= test_end:
            break
        yield np.arange(0, train_end), np.arange(train_end, test_end)


# ---------------------------------------------------------------------------
# Training & prediction
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    test_dates: pd.DatetimeIndex
    actual_prices: np.ndarray
    pred_prices: np.ndarray
    lower_prices: np.ndarray
    upper_prices: np.ndarray
    direction_proba: np.ndarray
    direction_actual: np.ndarray
    metrics: dict
    last_close: float
    last_date: pd.Timestamp
    next_date: pd.Timestamp
    next_price: float
    next_lower: float
    next_upper: float
    next_proba_up: float
    per_model_returns: dict = field(default_factory=dict)
    feature_importance: dict = field(default_factory=dict)


def _conformal_halfwidth(residuals: np.ndarray, alpha: float = 0.05) -> float:
    """Symmetric conformal interval half-width from absolute residuals."""
    if residuals.size == 0:
        return 0.0
    return float(np.quantile(np.abs(residuals), 1 - alpha))


def train_and_predict(
    ticker: str,
    period: str = "5y",
    use_macro: bool = True,
    horizon: int = 1,
) -> dict:
    raw = fetch_history(ticker, period=period)
    macro = fetch_macro(period=period) if use_macro else None
    df = add_features(raw, macro=macro)

    if len(df) < 150:
        raise ValueError(f"Not enough history for '{ticker}' ({len(df)} usable rows).")

    feat_cols = feature_columns(df)
    X = df[feat_cols].values
    y_return = df["target_return"].values
    y_up = df["target_up"].values
    close = df["close"].values

    test_size = max(40, int(len(df) * 0.2))
    X_train, X_test = X[:-test_size], X[-test_size:]
    yr_train, yr_test = y_return[:-test_size], y_return[-test_size:]
    yc_train, yc_test = y_up[:-test_size], y_up[-test_size:]
    close_test = close[-test_size:]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Regression ensemble ----
    specs = available_regressors()
    per_model_returns: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}

    for spec in specs:
        m = spec.build()
        Xt = X_train_s if spec.needs_scaling else X_train
        Xe = X_test_s if spec.needs_scaling else X_test
        m.fit(Xt, yr_train)
        per_model_returns[spec.name] = m.predict(Xe)
        fitted[spec.name] = m

    # naive baseline = 0% return
    per_model_returns["naive_zero"] = np.zeros_like(yr_test)

    pred_returns = np.mean(list(per_model_returns.values()), axis=0)
    pred_prices = close_test * (1 + pred_returns)
    actual_prices = close_test * (1 + yr_test)

    # Walk-forward residuals on training tail to calibrate intervals (conformal)
    residual_returns = []
    for tr_idx, va_idx in walk_forward_indices(len(X_train), n_folds=5, min_train=int(len(X_train) * 0.5)):
        Xt_tr_raw, Xt_va_raw = X_train[tr_idx], X_train[va_idx]
        sc = StandardScaler().fit(Xt_tr_raw)
        Xt_tr_s, Xt_va_s = sc.transform(Xt_tr_raw), sc.transform(Xt_va_raw)
        fold_preds = []
        for spec in specs:
            m = spec.build()
            Xt = Xt_tr_s if spec.needs_scaling else Xt_tr_raw
            Xe = Xt_va_s if spec.needs_scaling else Xt_va_raw
            m.fit(Xt, yr_train[tr_idx])
            fold_preds.append(m.predict(Xe))
        fold_preds.append(np.zeros(len(va_idx)))
        ensemble = np.mean(fold_preds, axis=0)
        residual_returns.append(yr_train[va_idx] - ensemble)
    residual_returns = np.concatenate(residual_returns) if residual_returns else np.array([])
    halfwidth_return = _conformal_halfwidth(residual_returns, alpha=0.05)

    lower_prices = close_test * (1 + pred_returns - halfwidth_return)
    upper_prices = close_test * (1 + pred_returns + halfwidth_return)

    mae = float(mean_absolute_error(actual_prices, pred_prices))
    rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
    r2 = float(r2_score(actual_prices, pred_prices))

    # ---- Direction classifier ----
    clf = make_rf_classifier()
    clf.fit(X_train_s, yc_train)
    proba = clf.predict_proba(X_test_s)
    proba_up = proba[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(yc_test))
    pred_up = (proba_up > 0.5).astype(int)
    direction_acc = float(accuracy_score(yc_test, pred_up))

    # ---- Forward forecast (multi-day, recursive, regression ensemble only) ----
    last_close = float(df["close"].iloc[-1])
    last_date = df.index[-1]

    forecast_dates: list[pd.Timestamp] = []
    forecast_prices: list[float] = []
    forecast_lower: list[float] = []
    forecast_upper: list[float] = []
    rolling_close = last_close
    rolling_features = df[feat_cols].iloc[[-1]].values.copy()
    feature_min = X_train.min(axis=0)
    feature_max = X_train.max(axis=0)
    return_idx = feat_cols.index("return") if "return" in feat_cols else None
    lag_indices = {
        lag: feat_cols.index(f"return_lag_{lag}") if f"return_lag_{lag}" in feat_cols else None
        for lag in (1, 2, 3, 5)
    }

    for step in range(horizon):
        clipped = np.clip(rolling_features, feature_min, feature_max)
        scaled = scaler.transform(clipped)
        step_returns = [
            (m.predict(scaled if spec.needs_scaling else clipped)[0])
            for spec, m in zip(specs, [fitted[s.name] for s in specs])
        ]
        step_returns.append(0.0)
        ensemble_return = float(np.mean(step_returns))
        new_price = rolling_close * (1 + ensemble_return)

        date = last_date + pd.tseries.offsets.BDay(step + 1)
        forecast_dates.append(date)
        forecast_prices.append(new_price)
        # Interval widens with sqrt(horizon) for a random-walk-ish drift in residuals
        scale = halfwidth_return * np.sqrt(step + 1)
        forecast_lower.append(rolling_close * (1 + ensemble_return - scale))
        forecast_upper.append(rolling_close * (1 + ensemble_return + scale))

        rolling_close = new_price
        if return_idx is not None:
            old_returns = {lag: rolling_features[0, idx] if idx is not None else 0.0
                           for lag, idx in lag_indices.items()}
            rolling_features[0, return_idx] = ensemble_return
            for lag, idx in lag_indices.items():
                if idx is None:
                    continue
                if lag == 1:
                    rolling_features[0, idx] = old_returns[1] if 1 in old_returns else 0.0
                else:
                    prev_lag = lag - 1
                    if prev_lag in old_returns:
                        rolling_features[0, idx] = old_returns[prev_lag]

    # Direction probability for the immediate next day
    last_scaled = scaler.transform(np.clip(df[feat_cols].iloc[[-1]].values, feature_min, feature_max))
    next_proba_up = float(clf.predict_proba(last_scaled)[0, list(clf.classes_).index(1)])

    # Feature importances (averaged across regressors that expose them)
    importances = {}
    for spec in specs:
        m = fitted[spec.name]
        if hasattr(m, "feature_importances_"):
            importances[spec.name] = dict(zip(feat_cols, m.feature_importances_.tolist()))

    return {
        "ticker": ticker.upper(),
        "history": raw,
        "feature_columns": feat_cols,
        "test_dates": df.index[-test_size:],
        "actual_prices": actual_prices,
        "pred_prices": pred_prices,
        "lower_prices": lower_prices,
        "upper_prices": upper_prices,
        "direction_proba": proba_up,
        "direction_actual": yc_test,
        "metrics": {
            "mae": mae, "rmse": rmse, "r2": r2,
            "direction_accuracy": direction_acc,
            "ci_halfwidth_return": halfwidth_return,
            "horizon": horizon,
            "models": [s.name for s in specs] + ["naive_zero"],
        },
        "last_close": last_close,
        "last_date": last_date,
        "next_date": forecast_dates[0] if forecast_dates else last_date,
        "next_price": forecast_prices[0] if forecast_prices else last_close,
        "next_lower": forecast_lower[0] if forecast_lower else last_close,
        "next_upper": forecast_upper[0] if forecast_upper else last_close,
        "next_proba_up": next_proba_up,
        "forecast_dates": forecast_dates,
        "forecast_prices": forecast_prices,
        "forecast_lower": forecast_lower,
        "forecast_upper": forecast_upper,
        "per_model_test_returns": per_model_returns,
        "feature_importance": importances,
    }
