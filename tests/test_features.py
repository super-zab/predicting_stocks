"""Smoke tests for feature engineering — no network required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict import (
    PRICE_FEATURES, INTRADAY_FEATURES, SEASONAL_FEATURES,
    add_features, feature_columns, walk_forward_indices,
)
import predict_classic


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    rets = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def test_advanced_features_no_nan_or_inf():
    df = add_features(_synthetic_ohlcv())
    cols = feature_columns(df)
    arr = df[cols].values
    assert np.isfinite(arr).all(), "Features contain NaN or inf"
    assert "target_return" in df.columns
    assert "target_up" in df.columns
    assert df["target_up"].isin([0, 1]).all()


def test_advanced_features_expected_columns_present():
    df = add_features(_synthetic_ohlcv())
    for col in PRICE_FEATURES + INTRADAY_FEATURES + SEASONAL_FEATURES:
        assert col in df.columns, f"missing {col}"


def test_classic_features_no_nan_or_inf():
    df = predict_classic.add_features(_synthetic_ohlcv())
    arr = df[predict_classic.FEATURES].values
    assert np.isfinite(arr).all()
    assert "target_return" in df.columns


def test_walk_forward_folds_are_disjoint_and_ordered():
    folds = list(walk_forward_indices(500, n_folds=5, min_train=200))
    assert len(folds) > 0
    for tr, te in folds:
        assert tr.max() < te.min(), "Train indices must precede test indices"
        assert len(np.intersect1d(tr, te)) == 0


def test_advanced_train_and_predict_end_to_end(monkeypatch):
    import predict
    monkeypatch.setattr(predict, "fetch_history", lambda t, period="5y": _synthetic_ohlcv(500))
    monkeypatch.setattr(predict, "fetch_macro", lambda period="5y": None)
    result = predict.train_and_predict("FAKE", period="2y", horizon=3)
    assert result["ticker"] == "FAKE"
    assert len(result["forecast_prices"]) == 3
    assert 0 <= result["next_proba_up"] <= 1
    for k in ("mae", "rmse", "r2", "direction_accuracy"):
        assert k in result["metrics"]


def test_classic_train_and_predict_end_to_end(monkeypatch):
    monkeypatch.setattr(predict_classic, "fetch_history",
                        lambda t, period="5y": _synthetic_ohlcv(500))
    result = predict_classic.train_and_predict("FAKE", period="2y")
    assert result["ticker"] == "FAKE"
    assert len(result["forecast_prices"]) == 1
    assert "mae" in result["metrics"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
