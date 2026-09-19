"""
app.py  —  Streamlit web app for Uber Traffic Volume Predictor
           Junctions 1-4 | Date + Hour input | LightGBM backend
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import plotly.graph_objects as go
import plotly.express as px

from src.predict import predict, predict_day

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h1 { font-size: 2.8rem; margin: 0; color: #1a1a2e; }
    .metric-card p  { color: #555; margin: 0; font-size: 0.9rem; }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #1976d2;
        border-radius: 4px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        color: #1a3a5c;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 4px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        color: #5a4000;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Uber_logo_2018.png",
             width=100)
    st.title("Traffic Predictor")
    st.markdown("---")

    st.subheader("Input Parameters")

    junction = st.selectbox(
        "Road Junction",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Junction {x}",
        help="Four monitored road junctions from the dataset (2015–2017)"
    )

    selected_date = st.date_input(
        "Date",
        value=date.today(),
        min_value=date(2015, 1, 1),
        help="Any date — the model uses day-of-week, month, hour as features"
    )

    hour = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=8,
        format="%d:00",
        help="0 = midnight, 8 = 8 AM, 17 = 5 PM"
    )

    st.markdown("---")
    st.markdown("**Advanced options**")
    show_chart    = st.checkbox("Show 24-hour forecast chart", value=True)
    show_heatmap  = st.checkbox("Show weekly pattern heatmap", value=False)

    st.markdown("---")
    st.caption("Model: LightGBM (refined) | R² = 0.9699 | MAE = 3.28")
    st.caption("Data: 2015–2017 road junction traffic")

    predict_btn = st.button("🔮 Predict Traffic Volume", type="primary", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚦 Junction Traffic Volume Predictor")
st.markdown(
    "Predicts hourly vehicle count at road junctions using a **LightGBM** model "
    "trained on 2015–2017 historical data (48,120 hourly records across 4 junctions)."
)

# ── Junction info cards ───────────────────────────────────────────────────────
JUNCTION_INFO = {
    1: {"records": "14,592", "avg_vehicles": "~28/hr",  "peak": "8–9 AM, 5–6 PM"},
    2: {"records": "14,592", "avg_vehicles": "~22/hr",  "peak": "8–9 AM, 4–5 PM"},
    3: {"records": "14,592", "avg_vehicles": "~8/hr",   "peak": "9–11 AM"},
    4: {"records": "4,344",  "avg_vehicles": "~12/hr",  "peak": "7–9 AM"},
}

col1, col2, col3 = st.columns(3)
with col1:
    info = JUNCTION_INFO[junction]
    st.markdown(f"""
    <div class="info-box">
      <b>Junction {junction}</b><br>
      Records: {info['records']} hourly entries<br>
      Typical volume: {info['avg_vehicles']}<br>
      Peak hours: {info['peak']}
    </div>""", unsafe_allow_html=True)

with col2:
    dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = selected_date.weekday()
    tod = "Night" if hour <= 5 else "Morning" if hour <= 11 else \
          "Afternoon" if hour <= 17 else "Evening" if hour <= 20 else "Night"
    rush = "Yes" if (7 <= hour <= 9 or 17 <= hour <= 19) else "No"
    st.markdown(f"""
    <div class="info-box">
      <b>Selected period</b><br>
      {selected_date.strftime('%d %b %Y')} ({dow_names[dow]})<br>
      {hour:02d}:00 — {tod} | Rush hour: {rush}<br>
      Weekend: {"Yes" if dow >= 5 else "No"}
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="warning-box">
      <b>Note on lag features</b><br>
      Lag (1h–168h) and rolling window features are set to 0 at inference
      since no live traffic history is available. Accuracy is slightly
      lower than the training R² = 0.97.
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn or "last_pred" in st.session_state:

    try:
        volume, model_name = predict(junction, selected_date, hour)

        # Store in session state so results persist across sidebar interactions
        st.session_state["last_pred"] = {
            "volume": volume, "model": model_name,
            "junction": junction, "date": selected_date, "hour": hour,
        }

        lp = st.session_state["last_pred"]

        # ── Result metric ─────────────────────────────────────────────────────
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <p>Predicted vehicles</p>
              <h1>{lp['volume']:.0f}</h1>
              <p>vehicles / hour at Junction {lp['junction']}</p>
            </div>""", unsafe_allow_html=True)

        with c2:
            traffic_level = (
                "Low" if lp["volume"] < 15 else
                "Moderate" if lp["volume"] < 35 else
                "High" if lp["volume"] < 60 else "Very High"
            )
            color = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Very High": "🔴"}[traffic_level]
            st.markdown(f"""
            <div class="metric-card">
              <p>Traffic level</p>
              <h1>{color}</h1>
              <p>{traffic_level}</p>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
              <p>Model used</p>
              <h1 style="font-size:1.3rem;padding-top:0.6rem">{lp['model']}</h1>
              <p>LightGBM (refined)</p>
            </div>""", unsafe_allow_html=True)

        st.success(
            f"Prediction for **Junction {lp['junction']}** on "
            f"**{lp['date'].strftime('%d %b %Y')}** at **{lp['hour']:02d}:00** → "
            f"**{lp['volume']:.0f} vehicles/hour**"
        )

        # ── 24-hour forecast chart ────────────────────────────────────────────
        if show_chart:
            st.subheader(f"24-Hour Forecast — Junction {lp['junction']} on {lp['date'].strftime('%d %b %Y')}")

            with st.spinner("Generating 24-hour forecast..."):
                hourly_preds = predict_day(lp["junction"], lp["date"])

            hours = list(range(24))
            hour_labels = [f"{h:02d}:00" for h in hours]

            # Rush hour bands
            fig = go.Figure()

            # Rush hour shading
            for rh_start, rh_end in [(7, 9), (17, 19)]:
                fig.add_vrect(x0=rh_start, x1=rh_end,
                              fillcolor="rgba(255,165,0,0.12)",
                              layer="below", line_width=0,
                              annotation_text="Rush", annotation_position="top left",
                              annotation_font_size=10)

            # Prediction line
            fig.add_trace(go.Scatter(
                x=hours, y=hourly_preds,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#1976d2", width=2.5),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(25,118,210,0.08)",
            ))

            # Highlight selected hour
            fig.add_trace(go.Scatter(
                x=[lp["hour"]], y=[lp["volume"]],
                mode="markers",
                name="Selected hour",
                marker=dict(color="#e53935", size=12, symbol="diamond"),
            ))

            fig.update_layout(
                xaxis=dict(title="Hour", tickvals=hours, ticktext=hour_labels,
                           tickangle=-45),
                yaxis=dict(title="Vehicles / hour"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
                margin=dict(l=40, r=20, t=30, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Orange bands = rush hours (7–9 AM, 5–7 PM). Red diamond = your selected hour.")

        # ── Weekly heatmap ────────────────────────────────────────────────────
        if show_heatmap:
            st.subheader(f"Weekly Pattern Heatmap — Junction {lp['junction']}")

            with st.spinner("Building weekly heatmap (168 predictions)..."):
                days   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                matrix = []
                # Use a reference Monday
                from datetime import timedelta
                ref_monday = lp["date"] - timedelta(days=lp["date"].weekday())
                for d_offset in range(7):
                    day_date = ref_monday + timedelta(days=d_offset)
                    row = [predict(lp["junction"], day_date, h)[0] for h in range(24)]
                    matrix.append(row)

            fig2 = px.imshow(
                matrix,
                labels=dict(x="Hour", y="Day", color="Vehicles/hr"),
                x=[f"{h:02d}:00" for h in range(24)],
                y=days,
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig2.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=40),
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info(
            "Make sure your `.pkl` model files are inside the `models/` folder in the project root. "
            "Expected: `FINAL_lgb_junction1.pkl` … `FINAL_lgb_junction4.pkl` "
            "or `FINAL_lightgbm_model.pkl`."
        )

else:
    st.info("Configure inputs in the sidebar and click **Predict Traffic Volume** to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built by Sayed Shoaib | LightGBM model trained on 2015–2017 junction traffic data | "
    "R² = 0.9699 | MAE = 3.28 vehicles/hour"
)
