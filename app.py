import streamlit as st
import pandas as pd
import json
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime
import time

# Application Configurations
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Define Shared File Path for Inter-Process Communication (IPC)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DATA_FILE = os.path.join(BASE_DIR, 'data', 'shared_data.json')

def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'rf_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def fetch_latest_data():
    """Reads the latest sensor data from the shared JSON file."""
    try:
        if os.path.exists(SHARED_DATA_FILE):
            with open(SHARED_DATA_FILE, 'r') as f:
                data = json.load(f)
            return data
    except Exception:
        pass
    return None

@st.cache_resource
def get_sensor_data_store():
    # Central storage for accumulating historical data (up to 50 items)
    return {"history": []}

store = get_sensor_data_store()
local_model = load_model()
model_status = local_model is not None

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# UI Layout Start
st.title("🏭 Predictive Machine Maintenance System")
st.markdown("Real-time telemetry from the *(Simulator)* is analyzed dynamically via a **Random Forest** AI Model.")

col1, col2 = st.columns([1, 4])

with col1:
    st.header("Control Panel")
    
    if st.button("▶️ Start Live Tracking", type="primary"):
        st.session_state.is_running = True
        
    if st.button("⏹️ Stop Tracking"):
        st.session_state.is_running = False
        
    if st.session_state.is_running:
        st.success("Live Tracking Active. Awaiting data flow...")
    else:
        st.warning("Tracking stopped.")
        
    st.markdown("---")
    st.markdown("**Machine Learning Model:** Status")
    if model_status:
        st.success("✅ Loaded (Random Forest)")
    else:
        st.error("❌ Not Found")

with col2:
    if st.session_state.is_running:
        
        # Read the latest data written by the simulator
        latest_sensor_data = fetch_latest_data()
        
        if latest_sensor_data is not None:
            # Check if this data is new to avoid adding duplicate ticks
            is_new_data = False
            if len(store["history"]) == 0 or store["history"][-1].get("Timestamp_raw") != latest_sensor_data.get("Timestamp_raw"):
                 is_new_data = True
                 
            if is_new_data:
                 # Live Machine Learning Prediction
                 prob = 0.0
                 if local_model is not None:
                     # Create DataFrame matching the model training features excluding categorical for now if needed, 
                     # but pipeline handles 'Type' since it was trained with it.
                     df_current = pd.DataFrame([latest_sensor_data])
                     # Drop the Timestamp_raw before prediction
                     if "Timestamp_raw" in df_current.columns:
                         df_current = df_current.drop("Timestamp_raw", axis=1)
                         
                     prediction_probs = local_model.predict_proba(df_current)
                     prob = prediction_probs[0][1] * 100
                 
                 # Enrich data
                 latest_sensor_data["Timestamp"] = datetime.now().strftime("%H:%M:%S")
                 latest_sensor_data["Failure_Prob"] = prob
                 
                 store["history"].append(latest_sensor_data)
                 # Keep only last 50 data points for UI fluidity
                 if len(store["history"]) > 50:
                     store["history"].pop(0)

        current_history = list(store["history"])
        
        if len(current_history) > 0:
            history_df = pd.DataFrame(current_history)
            last_data = current_history[-1]
            
            metric_cols = st.columns(5)
            metric_cols[0].metric("Air Temp.", f"{last_data['Air temperature']} K")
            metric_cols[1].metric("Process Temp.", f"{last_data['Process temperature']} K")
            metric_cols[2].metric("Rotational Speed", f"{last_data['Rotational speed']} RPM")
            metric_cols[3].metric("Torque", f"{last_data['Torque']} Nm")
            metric_cols[4].metric("Tool Wear", f"{last_data['Tool wear']} Min")
            
            st.markdown("### Failure Probability and Real-Time Architect State")
            prob_val = last_data['Failure_Prob']
            
            if prob_val < 30:
                st.success("✅ SYSTEM OPERATING NORMALLY")
            elif prob_val < 70:
                st.warning("⚠️ WARNING: ABNORMAL TELEMETRY DETECTED")
            else:
                st.error("🚨 CRITICAL MACHINE FAILURE RISK! PLEASE STOP THE SYSTEM!")
                
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Instant Failure Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("### Real-Time Parameter Telemetry")
            chart_cols = st.columns(2)
            
            fig_trq = go.Figure()
            fig_trq.add_trace(go.Scatter(x=history_df["Timestamp"], y=history_df["Torque"], mode='lines', name='Torque (Nm)', line=dict(color='orange')))
            fig_trq.update_layout(title="Torque Distribution Over Time", height=300, margin=dict(l=0, r=0, t=30, b=0))
            chart_cols[0].plotly_chart(fig_trq, use_container_width=True)
            
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=history_df["Timestamp"], y=history_df["Process temperature"], mode='lines', name='Process Temp (K)', line=dict(color='red')))
            fig_temp.update_layout(title="Temperature Distribution Over Time", height=300, margin=dict(l=0, r=0, t=30, b=0))
            chart_cols[1].plotly_chart(fig_temp, use_container_width=True)
            
        else:
            st.info("Awaiting Streamlit data stream. Please make sure `simulator.py` is running.")
            
        # Refresh delay to rerender UI smoothly
        time.sleep(1)
        st.rerun()
    else:
        st.info("Click 'Start Live Tracking' to begin monitoring.")
