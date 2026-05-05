"""Streamlit UI: pick a ticker, predict next close, animate the chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predict import train_and_predict


st.set_page_config(page_title="Stock Close Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #0b0d12; }
      .block-container { padding-top: 2rem; max-width: 1200px; }
      h1, h2, h3, h4, label, p { color: #e6e8eb !important; }
      div[data-testid="stMetricValue"] { color: #e6e8eb; font-weight: 600; }
      .hero {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
      }
      .hero-label { font-size: 0.95rem; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.55; }
      .hero-price { font-size: 4.2rem; font-weight: 700; line-height: 1.1; margin-top: 0.3rem; }
      .hero-delta { font-size: 1.4rem; font-weight: 500; opacity: 0.95; }
      .pulse-up { color: #2ecc71; text-shadow: 0 0 18px rgba(46,204,113,0.45); }
      .pulse-down { color: #ff5a5f; text-shadow: 0 0 18px rgba(255,90,95,0.45); }
      .stButton > button {
        border-radius: 999px; border: 1px solid #2a2f3a;
        background: #11141b; color: #e6e8eb; font-weight: 600;
        padding: 0.55rem 1.4rem;
      }
      .stButton > button:hover { border-color: #3d4554; }
    </style>
    """,
    unsafe_allow_html=True,
)


PRESETS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "AMD", "NFLX", "SPY"]


st.markdown("# :chart_with_upwards_trend: Stock Close Predictor")
st.caption("Random Forest on technical features. Educational use only — not financial advice.")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    choice = st.selectbox("Company", PRESETS + ["Other..."], index=1)
    if choice == "Other...":
        ticker = st.text_input("Ticker symbol", value="").strip().upper()
    else:
        ticker = choice
with c2:
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
with c3:
    st.write("")
    st.write("")
    run = st.button(":crystal_ball: Predict", use_container_width=True)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_predict(ticker: str, period: str):
    return train_and_predict(ticker, period=period)


def build_animated_figure(result: dict, color: str, fill_rgba: str) -> go.Figure:
    history = result["history"]
    test_dates = list(result["test_dates"])
    pred = list(result["pred_prices"])
    lower = list(result["lower_prices"])
    upper = list(result["upper_prices"])

    next_date = result["next_date"]
    next_price = result["next_price"]
    next_lower = result["next_lower"]
    next_upper = result["next_upper"]

    context_start = test_dates[0] - pd.Timedelta(days=120)
    ctx = history.loc[history.index >= context_start]

    actual_x = list(ctx.index)
    actual_y = list(ctx["close"].values)

    full_pred_dates = test_dates + [next_date]
    full_pred = pred + [next_price]
    full_lower = lower + [next_lower]
    full_upper = upper + [next_upper]

    n = len(full_pred_dates)
    frames = []
    for i in range(2, n + 1):
        d = full_pred_dates[:i]
        p = full_pred[:i]
        lo = full_lower[:i]
        up = full_upper[:i]
        ribbon_x = list(d) + list(d)[::-1]
        ribbon_y = list(up) + list(lo)[::-1]

        marker_x = [d[-1]]
        marker_y = [p[-1]]

        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter(x=actual_x, y=actual_y),
                    go.Scatter(x=ribbon_x, y=ribbon_y),
                    go.Scatter(x=d, y=p),
                    go.Scatter(x=marker_x, y=marker_y),
                ],
            )
        )

    init = frames[0]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=actual_x, y=actual_y, name="Actual",
                line=dict(color="#9aa0a6", width=2),
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=init.data[1].x, y=init.data[1].y,
                fill="toself", fillcolor=fill_rgba,
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", hoverinfo="skip", showlegend=False,
            ),
            go.Scatter(
                x=init.data[2].x, y=init.data[2].y, name="Predicted",
                line=dict(color=color, width=3),
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=init.data[3].x, y=init.data[3].y, name="",
                mode="markers",
                marker=dict(color=color, size=12, line=dict(color="white", width=2)),
                hoverinfo="skip", showlegend=False,
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0b0d12",
        paper_bgcolor="#0b0d12",
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, color="#8b93a1"),
        yaxis=dict(title="Price (USD)", gridcolor="#1a1f2a", color="#8b93a1", zeroline=False),
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"),
        updatemenus=[
            dict(
                type="buttons", showactive=False, x=0.02, y=1.12, xanchor="left",
                bgcolor="#11141b", bordercolor="#2a2f3a",
                buttons=[
                    dict(
                        label="Replay",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=45, redraw=True),
                                fromcurrent=False,
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                ],
            )
        ],
    )
    return fig


if run:
    if not ticker:
        st.warning("Please enter a ticker.")
        st.stop()

    with st.spinner(f"Training model on {ticker}..."):
        try:
            result = cached_predict(ticker, period)
        except Exception as e:
            st.error(f"Could not predict for '{ticker}': {e}")
            st.stop()

    last_close = result["last_close"]
    next_price = result["next_price"]
    going_up = next_price >= last_close
    color = "#2ecc71" if going_up else "#ff5a5f"
    fill_rgba = "rgba(46,204,113,0.15)" if going_up else "rgba(255,90,95,0.15)"
    arrow = "&#9650;" if going_up else "&#9660;"
    pct = (next_price - last_close) / last_close * 100
    pulse_class = "pulse-up" if going_up else "pulse-down"

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-label">{result['ticker']} &middot; predicted next close
              ({result['next_date'].strftime('%b %d, %Y')})</div>
          <div class="hero-price {pulse_class}">{arrow} ${next_price:,.2f}</div>
          <div class="hero-delta {pulse_class}">{pct:+.2f}% &nbsp;vs last close ${last_close:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = build_animated_figure(result, color, fill_rgba)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    m = result["metrics"]
    k1, k2, k3 = st.columns(3)
    k1.metric("MAE (test)", f"${m['mae']:.2f}")
    k2.metric("RMSE (test)", f"${m['rmse']:.2f}")
    k3.metric("R² (test)", f"{m['r2']:.3f}")

    st.markdown(
        "<div style='opacity:0.55; font-size:0.85rem; margin-top:1rem;'>"
        "The shaded ribbon is a 95% interval from disagreement across the forest's trees. "
        "Markets are noisy — treat the prediction as a directional hint, nothing more."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div style='opacity:0.55; padding:2rem 0;'>"
        "Pick a ticker and hit <b>Predict</b>. The chart animates the model's "
        "out-of-sample predictions, then reveals tomorrow's forecast — green if higher, red if lower."
        "</div>",
        unsafe_allow_html=True,
    )
