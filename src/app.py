import streamlit as st
import pandas as pd
import json
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DATA_FILE = os.path.join(BASE_DIR, "data", "shared_data.json")
MODEL_PATH       = os.path.join(BASE_DIR, "models", "rf_model.joblib")

# ── Load Model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ── Session-state defaults ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_ts" not in st.session_state:
    st.session_state.last_ts = None

# ── Helper: read latest JSON written by simulator ─────────────────────────────
def read_sensor_file():
    try:
        if os.path.exists(SHARED_DATA_FILE):
            with open(SHARED_DATA_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏭 Predictive Machine Maintenance System")
st.markdown("Real-time telemetry from the *(Simulator)* is analyzed via a **Random Forest** AI Model.")

sidebar, main = st.columns([1, 4])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with sidebar:
    st.header("Control Panel")
    st.markdown("---")
    st.markdown("**ML Model**")
    if model:
        st.success("✅ Loaded (Random Forest)")
    else:
        st.error("❌ Not Found")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.info(
        "1. Run this dashboard\n"
        "2. Open **localhost:8505**\n"
        "3. Start `simulator.py` in a second terminal\n"
        "4. Data appears automatically ✅"
    )

# ── Main area ─────────────────────────────────────────────────────────────────
with main:
    raw = read_sensor_file()

    # Check freshness: data must be written in the last 3 seconds
    data_is_live = (
        raw is not None
        and time.time() - raw.get("Timestamp_raw", 0) < 3.0
    )

    if not data_is_live:
        st.info("⏳ Waiting for `simulator.py` to start sending data...")
    else:
        # ── Ingest new sample only if timestamp changed ────────────────────
        ts_raw = raw["Timestamp_raw"]
        if ts_raw != st.session_state.last_ts:
            st.session_state.last_ts = ts_raw

            # Prediction
            prob = 0.0
            if model is not None:
                df = pd.DataFrame([raw])
                df = df.drop(columns=["Timestamp_raw"], errors="ignore")
                prob = model.predict_proba(df)[0][1] * 100

            entry = {
                "Timestamp":          datetime.now().strftime("%H:%M:%S"),
                "Type":               raw["Type"],
                "Air temperature":    raw["Air temperature"],
                "Process temperature":raw["Process temperature"],
                "Rotational speed":   raw["Rotational speed"],
                "Torque":             raw["Torque"],
                "Tool wear":          raw["Tool wear"],
                "Failure_Prob":       prob,
            }
            st.session_state.history.append(entry)
            if len(st.session_state.history) > 50:
                st.session_state.history.pop(0)

        history = st.session_state.history
        if not history:
            st.info("⏳ First data point arriving...")
        else:
            df_hist  = pd.DataFrame(history)
            last     = history[-1]
            prob_val = last["Failure_Prob"]

            # ── Metrics ────────────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Air Temp",        f"{last['Air temperature']} K")
            c2.metric("Process Temp",    f"{last['Process temperature']} K")
            c3.metric("Rotational Speed",f"{last['Rotational speed']} RPM")
            c4.metric("Torque",          f"{last['Torque']} Nm")
            c5.metric("Tool Wear",       f"{last['Tool wear']} Min")

            # ── Alert banner ───────────────────────────────────────────────
            st.markdown("### Failure Probability")
            if prob_val < 30:
                st.success("✅ SYSTEM OPERATING NORMALLY")
            elif prob_val < 70:
                st.warning("⚠️ WARNING: ABNORMAL TELEMETRY DETECTED")
            else:
                st.error("🚨 CRITICAL FAILURE RISK — STOP THE SYSTEM!")

            # ── Gauge ──────────────────────────────────────────────────────
            fig_g = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = prob_val,
                title  = {"text": "Failure Probability (%)"},
                domain = {"x": [0, 1], "y": [0, 1]},
                gauge  = {
                    "axis":      {"range": [0, 100]},
                    "bar":       {"color": "darkblue"},
                    "steps":     [
                        {"range": [0,  30], "color": "lightgreen"},
                        {"range": [30, 70], "color": "yellow"},
                        {"range": [70,100], "color": "red"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4},
                                  "thickness": 0.75, "value": 90},
                },
            ))
            fig_g.update_layout(height=260, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

            # ── Line Charts ────────────────────────────────────────────────
            st.markdown("### Real-Time Telemetry")
            left, right = st.columns(2)

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=df_hist["Timestamp"], y=df_hist["Torque"],
                mode="lines", name="Torque (Nm)", line=dict(color="orange")
            ))
            fig_t.update_layout(title="Torque Over Time",
                                height=300, margin=dict(l=0,r=0,t=30,b=0))
            left.plotly_chart(fig_t, use_container_width=True)

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=df_hist["Timestamp"], y=df_hist["Process temperature"],
                mode="lines", name="Process Temp (K)", line=dict(color="red")
            ))
            fig_p.update_layout(title="Temperature Over Time",
                                height=300, margin=dict(l=0,r=0,t=30,b=0))
            right.plotly_chart(fig_p, use_container_width=True)

# ── Auto-refresh every 1 second ───────────────────────────────────────────────
time.sleep(1)
st.rerun()
