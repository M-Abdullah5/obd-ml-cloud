import time
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. PAGE CONFIG & THEME SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="AR Diagnostic Dashboard", layout="wide", page_icon="🏎️", initial_sidebar_state="expanded")

# Custom CSS for a sleek dark theme feel
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .sidebar .sidebar-content { background: #262730; }
    h1, h2, h3 { color: #00ffcc !important; }
    .stMetric label { color: #a1a1a1 !important; }
</style>
""", unsafe_allow_html=True)

FIREBASE_DB_URL = "https://arapp-feb0f-default-rtdb.firebaseio.com/"

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def add_breaks_for_gaps(df, threshold_seconds=5):
    """ Prevents Plotly from drawing straight lines across missing data periods """
    if df.empty: return df
    df = df.sort_values("timestamp")
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    gap_mask = df['time_diff'] > threshold_seconds
    
    gap_rows = []
    for idx, row in df[gap_mask].iterrows():
        gap_row = row.copy()
        for col in df.columns:
            if col not in ['timestamp', 'time_diff', 'device_id']:
                gap_row[col] = None
        gap_row['timestamp'] = row['timestamp'] - timedelta(seconds=1)
        gap_rows.append(gap_row)
        
    if gap_rows:
        df_gaps = pd.DataFrame(gap_rows)
        df_final = pd.concat([df, df_gaps], ignore_index=True).sort_values("timestamp")
        return df_final.drop(columns=['time_diff'])
    return df.drop(columns=['time_diff'])

@st.cache_data(ttl=3)
def get_devices():
    try:
        res = requests.get(f"{FIREBASE_DB_URL}live.json?shallow=true")
        if res.status_code == 200 and res.json():
            return list(res.json().keys())
    except: pass
    return []

@st.cache_data(ttl=1)
def get_live_data(device_id):
    try:
        res = requests.get(f"{FIREBASE_DB_URL}live/{device_id}.json")
        if res.status_code == 200: return res.json()
    except: pass
    return None

@st.cache_data(ttl=5)
def get_history_data(device_id):
    try:
        # Fetch latest 2000 records for the 10-minute window
        res = requests.get(f"{FIREBASE_DB_URL}history/{device_id}.json?orderBy=\"$key\"&limitToLast=2000")
        if res.status_code == 200 and res.json():
            records = list(res.json().values())
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure ALL 14 columns exist to prevent Plotly crashes
            expected_cols = ["RPM", "Speed", "CoolantTemp", "EngineLoad", "Voltage", 
                             "IntakeTemp", "MAF", "ThrottlePos", "OilTemp", "MAP", 
                             "FuelLevel", "STFT", "LTFT", "O2Voltage"]
            for col in expected_cols:
                if col not in df.columns: df[col] = 0.0
            return df
    except: pass
    return pd.DataFrame()

# ---------------------------------------------------------
# 3. SIDEBAR (FILLED WITH CONTEXT)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204094.png", width=80)
    st.title("Vehicle Profile")
    
    devices = get_devices()
    device_id = st.selectbox("Active Device", devices) if devices else None
    
    st.divider()
    st.markdown("### 🚘 Suzuki Alto 800")
    st.markdown("- **Engine:** F8D (796cc 3-Cylinder)")
    st.markdown("- **System:** Speed-Density (MAP)")
    st.markdown("- **Protocol:** CAN 500kbps 11-bit")
    st.divider()
    
    st.markdown("### 🤖 ML Architecture")
    st.markdown("- **Model:** Predictive Diagnostic Net v1")
    st.markdown("- **Target Classes:** 9 Subsystems")
    st.markdown("- **Update Rate:** 2Hz (500ms)")

# ---------------------------------------------------------
# 4. DATA FETCHING & STATUS LOGIC
# ---------------------------------------------------------
st.title("🚗 AR Telemetry Command Center")

if device_id:
    latest = get_live_data(device_id)
    df = get_history_data(device_id)
    
    if latest:
        # 🟢 FIX: Ignore timezones! Just check if the data has CHANGED recently.
        current_data_str = str(latest)
        
        if "last_data_str" not in st.session_state or st.session_state.last_data_str != current_data_str:
            st.session_state.last_data_str = current_data_str
            st.session_state.last_update_time = time.time()
            
        seconds_ago = time.time() - st.session_state.get("last_update_time", time.time())
        is_online = seconds_ago < 7 
    else:
        is_online = False
else:
    is_online = False
    latest = None
    df = pd.DataFrame()

# Status Banner
if is_online:
    st.success("🟢 **SYSTEM ONLINE** — Live Data Streaming Active")
else:
    st.error("🔴 **SYSTEM OFFLINE** — Connection Lost or Engine Off")

# ---------------------------------------------------------
# 5. ML ALERTS
# ---------------------------------------------------------
if latest and is_online:
    ml_status = latest.get("ml_status", "Healthy")
    ml_alert = latest.get("ml_alert", "None")
    
    if ml_status == "Critical":
        st.error(f"🚨 **CRITICAL ML ALERT:** {ml_alert}")
    elif ml_status == "Warning":
        st.warning(f"⚠️ **ML WARNING:** {ml_alert}")

st.divider()

# ---------------------------------------------------------
# 6. TABBED INTERFACE
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Live Metrics", "📈 10-Minute Telemetry", "📝 Raw Historical Data"])

# ================= TAB 1: LIVE METRICS =================
with tab1:
    st.subheader("Real-Time Engine Status")
    
    if latest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RPM", int(latest.get("RPM", 0)))
        c2.metric("Speed", f"{int(latest.get('Speed', 0))} km/h")
        c3.metric("Engine Load", f"{float(latest.get('EngineLoad', 0))} %")
        c4.metric("Throttle", f"{float(latest.get('ThrottlePos', 0))} %")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Coolant Temp", f"{float(latest.get('CoolantTemp', 0))} °C")
        c6.metric("Oil Temp", f"{float(latest.get('OilTemp', 0))} °C")
        c7.metric("Intake Temp", f"{float(latest.get('IntakeTemp', 0))} °C")
        c8.metric("Voltage", f"{float(latest.get('Voltage', 0))} V")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("MAP Pressure", f"{float(latest.get('MAP', 0))} kPa")
        c10.metric("MAF Airflow", f"{float(latest.get('MAF', 0))} g/s")
        c11.metric("STFT / LTFT", f"{float(latest.get('STFT', 0))}% / {float(latest.get('LTFT', 0))}%")
        c12.metric("O2 Sensor", f"{float(latest.get('O2Voltage', 0))} V")
    else:
        st.info("Awaiting connection to display metrics...")

# ================= TAB 2: GRAPHS (LAST 10 MINS) =================
with tab2:
    if not df.empty:
        # STRICT 10-MINUTE WINDOW CUTOFF
        ten_mins_ago = df["timestamp"].max() - timedelta(minutes=10)
        df_graphs = df[df["timestamp"] >= ten_mins_ago].copy()
        
        df_plot = add_breaks_for_gaps(df_graphs, threshold_seconds=5)
        
        # Helper to create styled dark-themed Plotly charts
        def create_chart(data, y_col, title, color):
            fig = px.line(data, x="timestamp", y=y_col, title=title)
            fig.update_traces(connectgaps=False, line_color=color, line_width=2)
            fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10), height=250)
            return fig

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(create_chart(df_plot, "RPM", "Engine RPM", "#FF4B4B"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "CoolantTemp", "Coolant Temp (°C)", "#FFA500"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "MAP", "MAP Pressure (kPa)", "#AB63FA"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "STFT", "Short Term Fuel Trim (%)", "#E2D9F3"), use_container_width=True)
            
        with g2:
            st.plotly_chart(create_chart(df_plot, "Speed", "Vehicle Speed (km/h)", "#00CC96"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "OilTemp", "Oil Temp (°C)", "#F4D03F"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "IntakeTemp", "Intake Temp (°C)", "#58D68D"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "LTFT", "Long Term Fuel Trim (%)", "#A569BD"), use_container_width=True)
            
        with g3:
            st.plotly_chart(create_chart(df_plot, "EngineLoad", "Engine Load (%)", "#636EFA"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "ThrottlePos", "Throttle Position (%)", "#1ABC9C"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "Voltage", "Battery Voltage (V)", "#F39C12"), use_container_width=True)
            st.plotly_chart(create_chart(df_plot, "O2Voltage", "O2 Sensor (V)", "#E74C3C"), use_container_width=True)
            
    else:
        st.info("No historical data available yet. Start the engine to generate graphs!")

# ================= TAB 3: TABULAR DATA =================
with tab3:
    st.subheader("Raw Database Logs")
    if not df.empty:
        # Display most recent first
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No logs found.")

# ---------------------------------------------------------
# 7. AUTO-REFRESH LOGIC
# ---------------------------------------------------------
if is_online:
    time.sleep(1) # Refresh fast when driving
    st.rerun()
else:
    time.sleep(3) # Refresh slowly when offline to save bandwidth
    st.rerun()
