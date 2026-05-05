"""Streamlit UI for the stock close predictor.

Two model engines are wired up side-by-side:
  - "Classique"  -> predict_classic.train_and_predict (original single Random Forest)
  - "Avance"     -> predict.train_and_predict (ensemble + macro + classifier + conformal)

Tabs: Predict, Backtest, Compare.
"""

from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import predict as predict_advanced
import predict_classic
from predict import (
    DataFetchError,
    InsufficientDataError,
    PredictionError,
    TickerNotFoundError,
)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


CACHE_DIR = os.path.join(os.path.dirname(__file__), ".prediction_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


st.set_page_config(page_title="Stock Close Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #0b0d12; }
      .block-container { padding-top: 1.6rem; max-width: 1280px; }
      h1, h2, h3, h4, label, p { color: #e6e8eb !important; }
      div[data-testid="stMetricValue"] { color: #e6e8eb; font-weight: 600; }
      .hero { text-align: center; padding: 0.8rem 0 0.4rem 0; }
      .hero-label { font-size: 0.9rem; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.55; }
      .hero-price { font-size: 4rem; font-weight: 700; line-height: 1.05; margin-top: 0.3rem; }
      .hero-delta { font-size: 1.3rem; font-weight: 500; opacity: 0.95; }
      .pulse-up { color: #2ecc71; text-shadow: 0 0 18px rgba(46,204,113,0.45); }
      .pulse-down { color: #ff5a5f; text-shadow: 0 0 18px rgba(255,90,95,0.45); }
      .stButton > button {
        border-radius: 999px; border: 1px solid #2a2f3a;
        background: #11141b; color: #e6e8eb; font-weight: 600;
        padding: 0.55rem 1.4rem;
      }
      .stButton > button:hover { border-color: #3d4554; }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] {
        background: #11141b; border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem; color: #9aa0a6;
      }
      .stTabs [aria-selected="true"] { background: #1a1f2a; color: #e6e8eb; }
    </style>
    """,
    unsafe_allow_html=True,
)


PRESET_GROUPS = {
    "US large caps": ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "AMD", "NFLX"],
    "Indices & ETFs": ["SPY", "QQQ", "DIA", "IWM", "VTI", "EEM"],
    "International": ["BABA", "TM", "ASML", "MC.PA", "AIR.PA", "OR.PA", "SAP", "TSM"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "FX": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"],
}
PRESETS = [t for group in PRESET_GROUPS.values() for t in group]
ENGINES = {
    "Classique (Random Forest seul)": predict_classic,
    "Avance (Ensemble + macro + classifier)": predict_advanced,
}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(engine_label: str, ticker: str, period: str, horizon: int, use_macro: bool) -> str:
    raw = f"{engine_label}|{ticker.upper()}|{period}|{horizon}|{use_macro}"
    return hashlib.md5(raw.encode()).hexdigest()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_predict(engine_label: str, ticker: str, period: str, horizon: int,
                   use_macro: bool, force_refresh: bool = False):
    """Streamlit memory cache + joblib disk cache (if available).

    `force_refresh=True` skips both caches for this call. Streamlit's @cache_data
    keys on all positional args, so flipping force_refresh between True/False
    creates two distinct cache entries — exactly what we want.
    """
    key = _cache_key(engine_label, ticker, period, horizon, use_macro)
    disk_path = os.path.join(CACHE_DIR, f"{key}.joblib")
    if not force_refresh and HAS_JOBLIB and os.path.exists(disk_path):
        try:
            logger.info("Cache hit for %s (%s)", ticker, engine_label)
            return joblib.load(disk_path)
        except Exception as e:
            logger.warning("Disk cache read failed (%s); recomputing", e)

    logger.info("Training %s on %s (period=%s, horizon=%s, macro=%s, refresh=%s)",
                engine_label, ticker, period, horizon, use_macro, force_refresh)
    engine = ENGINES[engine_label]
    if engine is predict_advanced:
        result = engine.train_and_predict(ticker, period=period,
                                          horizon=horizon, use_macro=use_macro)
    else:
        result = engine.train_and_predict(ticker, period=period)

    if HAS_JOBLIB:
        try:
            joblib.dump(result, disk_path)
        except Exception as e:
            logger.warning("Disk cache write failed (%s)", e)
    return result


def show_prediction_error(ticker: str, exc: Exception) -> None:
    """Render a user-friendly message for a known prediction failure."""
    if isinstance(exc, TickerNotFoundError):
        st.error(f":mag: Ticker not found")
        st.info(str(exc))
    elif isinstance(exc, InsufficientDataError):
        st.warning(f":hourglass: Not enough history")
        st.info(str(exc))
    elif isinstance(exc, DataFetchError):
        st.error(f":satellite: Data provider error")
        st.info(str(exc))
        st.caption("If the issue persists, retry in a few seconds — yfinance can rate-limit.")
    elif isinstance(exc, PredictionError):
        st.error(str(exc))
    else:
        logger.exception("Unexpected prediction failure for %s", ticker)
        st.error(f"Unexpected error for '{ticker}': {exc}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _ribbon(x, lo, up):
    return list(x) + list(x)[::-1], list(up) + list(lo)[::-1]


def build_animated_line(result: dict, color: str, fill_rgba: str) -> go.Figure:
    history = result["history"]
    test_dates = list(result["test_dates"])
    pred = list(result["pred_prices"])
    lower = list(result["lower_prices"])
    upper = list(result["upper_prices"])

    forecast_dates = list(result.get("forecast_dates") or [result["next_date"]])
    forecast_prices = list(result.get("forecast_prices") or [result["next_price"]])
    forecast_lower = list(result.get("forecast_lower") or [result["next_lower"]])
    forecast_upper = list(result.get("forecast_upper") or [result["next_upper"]])

    context_start = test_dates[0] - pd.Timedelta(days=120)
    ctx = history.loc[history.index >= context_start]
    actual_x = list(ctx.index)
    actual_y = list(ctx["close"].values)

    full_dates = test_dates + forecast_dates
    full_pred = pred + forecast_prices
    full_lower = lower + forecast_lower
    full_upper = upper + forecast_upper

    n = len(full_dates)
    frames = []
    for i in range(2, n + 1):
        d = full_dates[:i]
        p = full_pred[:i]
        lo = full_lower[:i]
        up = full_upper[:i]
        rx, ry = _ribbon(d, lo, up)
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=actual_x, y=actual_y),
                go.Scatter(x=rx, y=ry),
                go.Scatter(x=d, y=p),
                go.Scatter(x=[d[-1]], y=[p[-1]]),
            ],
        ))

    init = frames[0]
    fig = go.Figure(
        data=[
            go.Scatter(x=actual_x, y=actual_y, name="Actual",
                       line=dict(color="#9aa0a6", width=2),
                       hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"),
            go.Scatter(x=init.data[1].x, y=init.data[1].y,
                       fill="toself", fillcolor=fill_rgba,
                       line=dict(color="rgba(0,0,0,0)"),
                       name="CI", hoverinfo="skip", showlegend=False),
            go.Scatter(x=init.data[2].x, y=init.data[2].y, name="Predicted",
                       line=dict(color=color, width=3),
                       hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"),
            go.Scatter(x=init.data[3].x, y=init.data[3].y, name="",
                       mode="markers",
                       marker=dict(color=color, size=12, line=dict(color="white", width=2)),
                       hoverinfo="skip", showlegend=False),
        ],
        frames=frames,
    )
    _style_chart(fig)
    return fig


def build_candlestick(result: dict, color: str) -> go.Figure:
    history = result["history"]
    test_dates = list(result["test_dates"])
    context_start = test_dates[0] - pd.Timedelta(days=120)
    ctx = history.loc[history.index >= context_start]

    forecast_dates = list(result.get("forecast_dates") or [result["next_date"]])
    forecast_prices = list(result.get("forecast_prices") or [result["next_price"]])

    fig = go.Figure(data=[
        go.Candlestick(
            x=ctx.index, open=ctx["open"], high=ctx["high"], low=ctx["low"], close=ctx["close"],
            name="OHLC", increasing_line_color="#2ecc71", decreasing_line_color="#ff5a5f",
        ),
        go.Scatter(
            x=[ctx.index[-1]] + forecast_dates,
            y=[float(ctx["close"].iloc[-1])] + forecast_prices,
            mode="lines+markers", name="Forecast",
            line=dict(color=color, width=3, dash="dot"),
            marker=dict(size=8, color=color, line=dict(color="white", width=2)),
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
        ),
    ])
    _style_chart(fig)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def _style_chart(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0b0d12",
        paper_bgcolor="#0b0d12",
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, color="#8b93a1"),
        yaxis=dict(title="Price (USD)", gridcolor="#1a1f2a", color="#8b93a1", zeroline=False),
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=1.12, xanchor="left",
            bgcolor="#11141b", bordercolor="#2a2f3a",
            buttons=[dict(label="Replay", method="animate",
                          args=[None, dict(frame=dict(duration=45, redraw=True),
                                           fromcurrent=False, mode="immediate",
                                           transition=dict(duration=0))])],
        )],
    )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _max_drawdown(curve: np.ndarray) -> float:
    """Worst peak-to-trough decline of an equity curve. Returns a negative number."""
    if len(curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(curve)
    drawdowns = curve / running_max - 1
    return float(drawdowns.min())


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe with rf=0. Returns 0 if std is zero."""
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def backtest_signal(result: dict, cost_bps: float = 0.0) -> dict:
    """Long/flat strategy: long on day t+1 when the model predicts an up move.

    cost_bps applies on every position change (entry and exit), expressed in basis
    points of notional (e.g. 10 = 0.10%).
    """
    actual = np.asarray(result["actual_prices"], dtype=float)
    pred = np.asarray(result["pred_prices"], dtype=float)
    history = result["history"]
    test_dates = list(result["test_dates"])
    closes_t = history["close"].reindex(test_dates).values.astype(float)
    realized_returns = (actual / closes_t) - 1
    pred_returns = (pred / closes_t) - 1
    signal = (pred_returns > 0).astype(float)

    # Cost is paid whenever the position changes between consecutive days.
    position_changes = np.abs(np.diff(np.concatenate([[0.0], signal])))
    costs = position_changes * (cost_bps / 10_000.0)

    strat_returns = signal * realized_returns - costs
    bh_curve = np.cumprod(1 + realized_returns)
    strat_curve = np.cumprod(1 + strat_returns)

    long_days_returns = strat_returns[signal == 1]
    wins = long_days_returns[long_days_returns > 0]
    losses = long_days_returns[long_days_returns < 0]
    win_rate = float(len(wins) / len(long_days_returns)) if len(long_days_returns) else 0.0
    profit_factor = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")

    return {
        "dates": test_dates,
        "buy_hold": bh_curve,
        "strategy": strat_curve,
        "strategy_returns": strat_returns,
        "n_long_days": int(signal.sum()),
        "n_trades": int(position_changes.sum()),
        "total_days": len(signal),
        "strategy_total_return": float(strat_curve[-1] - 1),
        "buy_hold_total_return": float(bh_curve[-1] - 1),
        "hit_rate": float(((pred_returns > 0) == (realized_returns > 0)).mean()),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe": _sharpe(strat_returns),
        "max_drawdown": _max_drawdown(strat_curve),
        "buy_hold_max_drawdown": _max_drawdown(bh_curve),
        "cost_bps": cost_bps,
    }


def build_backtest_chart(bt: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["dates"], y=(bt["buy_hold"] - 1) * 100,
                             name="Buy & Hold", line=dict(color="#9aa0a6", width=2)))
    fig.add_trace(go.Scatter(x=bt["dates"], y=(bt["strategy"] - 1) * 100,
                             name="Model long/flat", line=dict(color="#2ecc71", width=3)))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#0b0d12", paper_bgcolor="#0b0d12",
        height=420, margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, color="#8b93a1"),
        yaxis=dict(title="Cumulative return (%)", gridcolor="#1a1f2a", color="#8b93a1"),
        legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown("# :chart_with_upwards_trend: Stock Close Predictor")
st.caption("Educational. Not financial advice.")

with st.sidebar:
    st.markdown("### Settings")
    engine_label = st.radio(
        "Model engine",
        list(ENGINES.keys()),
        index=1,
        help="Classique = original single Random Forest. Avance = ensemble RF+LightGBM+naive, "
             "macro features, conformal intervals, direction classifier.",
    )
    period = st.selectbox("History window", ["1y", "2y", "5y", "10y"], index=2)
    horizon = st.slider("Forecast horizon (business days)", 1, 10, 5,
                        disabled=engine_label.startswith("Classique"),
                        help="Multi-day recursive forecast (advanced engine only).")
    use_macro = st.checkbox("Include macro features (VIX, S&P, 10Y)", value=True,
                            disabled=engine_label.startswith("Classique"))
    chart_mode = st.radio("Chart style", ["Line + animation", "Candlestick"], index=0)
    force_refresh = st.checkbox(
        "Refresh data on next run",
        value=False,
        help="Bypass both the memory and disk caches for the next prediction. "
             "Use this when intraday or end-of-day data has just updated.",
    )

    if HAS_JOBLIB and st.button("Clear disk cache", use_container_width=True):
        for f in os.listdir(CACHE_DIR):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except OSError:
                pass
        st.success("Cache cleared.")

tab_predict, tab_backtest, tab_compare = st.tabs(["Predict", "Backtest", "Compare"])

# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        group = st.selectbox("Asset class", list(PRESET_GROUPS.keys()) + ["Other..."],
                             index=0, key="predict_group")
    with c2:
        if group == "Other...":
            ticker = st.text_input("Ticker symbol (Yahoo Finance format)",
                                   value="", key="predict_other").strip().upper()
        else:
            ticker = st.selectbox("Ticker", PRESET_GROUPS[group], key="predict_ticker")
    with c3:
        st.write("")
        st.write("")
        run = st.button(":crystal_ball: Predict", use_container_width=True, key="predict_btn")

    if run:
        if not ticker:
            st.warning("Please enter a ticker.")
            st.stop()
        with st.spinner(f"[{engine_label}] Training on {ticker}..."):
            try:
                result = cached_predict(engine_label, ticker, period, horizon,
                                        use_macro, force_refresh=force_refresh)
            except Exception as e:
                show_prediction_error(ticker, e)
                st.stop()
        st.session_state["last_result"] = result
        st.session_state["last_engine"] = engine_label

    result = st.session_state.get("last_result")
    if result:
        last_close = result["last_close"]
        next_price = result["next_price"]
        going_up = next_price >= last_close
        color = "#2ecc71" if going_up else "#ff5a5f"
        fill_rgba = "rgba(46,204,113,0.15)" if going_up else "rgba(255,90,95,0.15)"
        arrow = "&#9650;" if going_up else "&#9660;"
        pct = (next_price - last_close) / last_close * 100
        pulse_class = "pulse-up" if going_up else "pulse-down"
        proba_up = result.get("next_proba_up", 0.5) * 100

        forecast_dates = result.get("forecast_dates") or [result["next_date"]]
        last_forecast_date = forecast_dates[-1]

        st.markdown(
            f"""
            <div class="hero">
              <div class="hero-label">{result['ticker']} &middot; predicted close
                  ({result['next_date'].strftime('%b %d')} ... {last_forecast_date.strftime('%b %d, %Y')})</div>
              <div class="hero-price {pulse_class}">{arrow} ${next_price:,.2f}</div>
              <div class="hero-delta {pulse_class}">{pct:+.2f}% &nbsp;vs ${last_close:,.2f}
                  &nbsp;&middot;&nbsp; P(up) = {proba_up:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if chart_mode.startswith("Candlestick"):
            fig = build_candlestick(result, color)
        else:
            fig = build_animated_line(result, color, fill_rgba)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        m = result["metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE (test)", f"${m['mae']:.2f}")
        k2.metric("RMSE (test)", f"${m['rmse']:.2f}")
        k3.metric("R^2 (test)", f"{m['r2']:.3f}")
        k4.metric("Direction acc.", f"{m.get('direction_accuracy', 0)*100:.1f}%")

        # CSV export: out-of-sample predictions and the forward forecast.
        export_test = pd.DataFrame({
            "date": result["test_dates"],
            "actual_close": result["actual_prices"],
            "predicted_close": result["pred_prices"],
            "lower_95": result["lower_prices"],
            "upper_95": result["upper_prices"],
            "kind": "test",
        })
        export_forecast = pd.DataFrame({
            "date": result.get("forecast_dates") or [result["next_date"]],
            "actual_close": np.nan,
            "predicted_close": result.get("forecast_prices") or [result["next_price"]],
            "lower_95": result.get("forecast_lower") or [result["next_lower"]],
            "upper_95": result.get("forecast_upper") or [result["next_upper"]],
            "kind": "forecast",
        })
        export_df = pd.concat([export_test, export_forecast], ignore_index=True)
        st.download_button(
            ":arrow_down: Download predictions (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{result['ticker']}_predictions.csv",
            mime="text/csv",
            use_container_width=False,
        )

        if result.get("feature_importance"):
            with st.expander("Feature importances"):
                rows = []
                for model_name, imps in result["feature_importance"].items():
                    for f, v in imps.items():
                        rows.append({"model": model_name, "feature": f, "importance": v})
                if rows:
                    imp_df = pd.DataFrame(rows).pivot_table(
                        index="feature", columns="model", values="importance", fill_value=0,
                    ).sort_values(by=list({r["model"] for r in rows})[0], ascending=False)
                    st.dataframe(imp_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Backtest tab
# ---------------------------------------------------------------------------
with tab_backtest:
    result = st.session_state.get("last_result")
    if not result:
        st.info("Run a prediction first to see the backtest.")
    else:
        bt_col1, bt_col2 = st.columns([2, 1])
        with bt_col1:
            cost_bps = st.slider(
                "Transaction cost (basis points per side)",
                min_value=0, max_value=50, value=5, step=1,
                help="Charged on every position change. 5 bps ~ 0.05% per entry/exit.",
            )
        with bt_col2:
            bt_mode = st.radio(
                "Backtest mode",
                ["Single split", "Walk-forward refit"],
                index=0, horizontal=False,
                help="Walk-forward retrains the model at each fold boundary across the "
                     "whole test window — more realistic than a single train/test split. "
                     "Advanced engine only; Classique falls back to single split.",
            )

        if bt_mode == "Walk-forward refit":
            try:
                with st.spinner("Walk-forward refit..."):
                    wf_result = predict_advanced.walk_forward_backtest(
                        result["ticker"], period=period, n_folds=5, use_macro=use_macro,
                    )
                bt = backtest_signal(wf_result, cost_bps=float(cost_bps))
                st.caption(
                    f"Walk-forward over {wf_result['metrics']['n_folds']} folds, "
                    f"{len(wf_result['test_dates'])} OOS days. "
                    f"Direction acc {wf_result['metrics']['direction_accuracy']*100:.1f}%."
                )
            except Exception as e:
                st.error(f"Walk-forward failed: {e}. Falling back to single split.")
                bt = backtest_signal(result, cost_bps=float(cost_bps))
        else:
            bt = backtest_signal(result, cost_bps=float(cost_bps))

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Strategy return", f"{bt['strategy_total_return']*100:+.2f}%",
                    delta=f"vs B&H {(bt['strategy_total_return']-bt['buy_hold_total_return'])*100:+.2f}%")
        r1c2.metric("Buy & hold", f"{bt['buy_hold_total_return']*100:+.2f}%")
        r1c3.metric("Sharpe (ann.)", f"{bt['sharpe']:.2f}")
        r1c4.metric("Max drawdown", f"{bt['max_drawdown']*100:.1f}%",
                    delta=f"B&H {bt['buy_hold_max_drawdown']*100:.1f}%", delta_color="inverse")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("Hit rate", f"{bt['hit_rate']*100:.1f}%")
        r2c2.metric("Win rate (long days)", f"{bt['win_rate']*100:.1f}%")
        pf = bt["profit_factor"]
        r2c3.metric("Profit factor", "inf" if pf == float("inf") else f"{pf:.2f}")
        r2c4.metric("Trades", f"{bt['n_trades']}",
                    delta=f"{bt['n_long_days']}/{bt['total_days']} days long")

        st.plotly_chart(build_backtest_chart(bt), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            f"Long/flat strategy: long on day t+1 when model predicts up. "
            f"Costs: {cost_bps} bps per position change. No slippage modeled. "
            "Out-of-sample test window only."
        )

# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------
with tab_compare:
    cmp_mode = st.radio(
        "Mode",
        ["Ranking (multi-ticker)", "Head-to-head (Classique vs Avance, single ticker)"],
        index=0, horizontal=True, key="cmp_mode",
    )

    if cmp_mode.startswith("Ranking"):
        st.markdown("Pick a few tickers to compare next-day forecasts side by side.")
        cmp_group = st.selectbox("Asset class", list(PRESET_GROUPS.keys()),
                                 index=0, key="cmp_group")
        cmp_tickers = st.multiselect(
            "Tickers", PRESET_GROUPS[cmp_group],
            default=PRESET_GROUPS[cmp_group][:3],
            key="cmp_tickers_select",
        )
        cmp_run = st.button("Compare", key="cmp_btn")

        if cmp_run and cmp_tickers:
            rows = []
            progress = st.progress(0.0)
            for i, t in enumerate(cmp_tickers):
                try:
                    r = cached_predict(engine_label, t, period, horizon, use_macro,
                                       force_refresh=force_refresh)
                    rows.append({
                        "Ticker": r["ticker"],
                        "Last close": r["last_close"],
                        "Next predicted": r["next_price"],
                        "Change %": (r["next_price"] - r["last_close"]) / r["last_close"] * 100,
                        "P(up)": r.get("next_proba_up", 0.5) * 100,
                        "MAE": r["metrics"]["mae"],
                        "Dir acc": r["metrics"].get("direction_accuracy", 0) * 100,
                    })
                except Exception as e:
                    rows.append({"Ticker": t.upper(), "Last close": None,
                                 "Next predicted": None, "Change %": None,
                                 "P(up)": None, "MAE": None, "Dir acc": None,
                                 "error": str(e)})
                progress.progress((i + 1) / len(cmp_tickers))
            progress.empty()

            df_cmp = pd.DataFrame(rows)

            def _style(v):
                if isinstance(v, (int, float)) and not pd.isna(v):
                    return f"color: {'#2ecc71' if v >= 0 else '#ff5a5f'}; font-weight:600;"
                return ""

            st.dataframe(
                df_cmp.style.applymap(_style, subset=["Change %"]).format({
                    "Last close": "${:,.2f}", "Next predicted": "${:,.2f}",
                    "Change %": "{:+.2f}%", "P(up)": "{:.0f}%",
                    "MAE": "${:.2f}", "Dir acc": "{:.1f}%",
                }),
                use_container_width=True,
            )

            plot_df = df_cmp.dropna(subset=["Change %"])
            if not plot_df.empty:
                fig = go.Figure(go.Bar(
                    x=plot_df["Ticker"], y=plot_df["Change %"],
                    marker_color=["#2ecc71" if v >= 0 else "#ff5a5f"
                                  for v in plot_df["Change %"]],
                    hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
                ))
                fig.update_layout(
                    template="plotly_dark", plot_bgcolor="#0b0d12", paper_bgcolor="#0b0d12",
                    height=360, margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(title="Predicted next-day move (%)", gridcolor="#1a1f2a"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        st.markdown("Run both engines on the same ticker and overlay their predictions.")
        h2h_group = st.selectbox("Asset class", list(PRESET_GROUPS.keys()),
                                 index=0, key="h2h_group")
        h2h_ticker = st.selectbox("Ticker", PRESET_GROUPS[h2h_group], key="h2h_ticker")
        h2h_run = st.button("Run head-to-head", key="h2h_btn")

        if h2h_run and h2h_ticker:
            try:
                with st.spinner(f"Running both engines on {h2h_ticker}..."):
                    classic_r = cached_predict("Classique (Random Forest seul)",
                                               h2h_ticker, period, 1, False,
                                               force_refresh=force_refresh)
                    advanced_r = cached_predict("Avance (Ensemble + macro + classifier)",
                                                h2h_ticker, period, horizon, use_macro,
                                                force_refresh=force_refresh)
            except Exception as e:
                show_prediction_error(h2h_ticker, e)
                st.stop()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(classic_r["test_dates"]), y=list(classic_r["actual_prices"]),
                name="Actual", line=dict(color="#9aa0a6", width=2),
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=list(classic_r["test_dates"]), y=list(classic_r["pred_prices"]),
                name="Classique", line=dict(color="#3b82f6", width=2.5),
                hovertemplate="Classique<br>%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=list(advanced_r["test_dates"]), y=list(advanced_r["pred_prices"]),
                name="Avance", line=dict(color="#2ecc71", width=2.5),
                hovertemplate="Avance<br>%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="#0b0d12", paper_bgcolor="#0b0d12",
                height=480, margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, color="#8b93a1"),
                yaxis=dict(title="Price (USD)", gridcolor="#1a1f2a", color="#8b93a1"),
                legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            mc = classic_r["metrics"]
            ma = advanced_r["metrics"]
            cmp_df = pd.DataFrame({
                "Classique": [mc["mae"], mc["rmse"], mc["r2"],
                              mc.get("direction_accuracy", 0) * 100,
                              classic_r["next_price"]],
                "Avance":    [ma["mae"], ma["rmse"], ma["r2"],
                              ma.get("direction_accuracy", 0) * 100,
                              advanced_r["next_price"]],
            }, index=["MAE ($)", "RMSE ($)", "R^2",
                      "Direction acc (%)", "Next predicted ($)"])
            st.dataframe(cmp_df.style.format("{:.3f}"), use_container_width=True)
