import time
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from datetime import datetime, timedelta

# 1. Config & Setup
st.set_page_config(page_title="OBD Dashboard", layout="wide", page_icon="🚗")
st.title("🚗 OBD Predictive Maintenance Dashboard")

FIREBASE_DB_URL = "https://arapp-feb0f-default-rtdb.firebaseio.com/"

# --- HELPER: Insert Gaps for Offline Periods ---
def add_breaks_for_gaps(df, threshold_seconds=15):
    if df.empty:
        return df
    
    df = df.sort_values("timestamp")
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    gap_mask = df['time_diff'] > threshold_seconds
    
    gap_rows = []
    for idx, row in df[gap_mask].iterrows():
        gap_row = row.copy()
        gap_row['RPM'] = None
        gap_row['Speed'] = None
        gap_row['CoolantTemp'] = None
        gap_row['EngineLoad'] = None
        gap_row['Voltage'] = None
        gap_row['IntakeTemp'] = None
        gap_row['MAF'] = None
        gap_row['ThrottlePos'] = None
        gap_row['OilTemp'] = None
        gap_row['MAP'] = None
        gap_row['FuelLevel'] = None
        gap_row['STFT'] = None
        gap_row['LTFT'] = None
        gap_row['O2Voltage'] = None
        gap_row['timestamp'] = row['timestamp'] - timedelta(seconds=1)
        gap_rows.append(gap_row)
    
    if gap_rows:
        df_gaps = pd.DataFrame(gap_rows)
        df_final = pd.concat([df, df_gaps], ignore_index=True)
        df_final = df_final.sort_values("timestamp")
        return df_final.drop(columns=['time_diff'])
    
    return df.drop(columns=['time_diff'])

def format_time_ago(seconds):
    seconds = int(seconds)
    if seconds < 60: return f"{seconds} seconds"
    elif seconds < 3600: return f"{seconds // 60} minutes"
    elif seconds < 86400: return f"{seconds // 3600} hours"
    else: return f"{seconds // 86400} days"

@st.cache_data(ttl=5) # Cache for 5 seconds to prevent Firebase spam
def get_devices():
    try:
        res = requests.get(f"{FIREBASE_DB_URL}live.json?shallow=true")
        if res.status_code == 200 and res.json():
            return list(res.json().keys())
    except:
        pass
    return []

@st.cache_data(ttl=2)
def get_live_data(device_id):
    try:
        res = requests.get(f"{FIREBASE_DB_URL}live/{device_id}.json")
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None

@st.cache_data(ttl=10)
def get_history_data(device_id):
    try:
        # Fetch latest 1000 records
        res = requests.get(f"{FIREBASE_DB_URL}history/{device_id}.json?orderBy=\"$key\"&limitToLast=1000")
        if res.status_code == 200 and res.json():
            data = res.json()
            # Convert dictionary of dicts to list of dicts
            records = list(data.values())
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except:
        pass
    return pd.DataFrame()

# 2. Sidebar & Filters
devices = get_devices()
device_id = None

with st.sidebar:
    st.header("Filters")
    if devices:
        device_id = st.selectbox("Device", devices)
    else:
        st.warning("No devices found online.")

# 3. Main Data Logic
if device_id:
    latest = get_live_data(device_id)
    df = get_history_data(device_id)
    
    if latest:
        now = datetime.now()
        last_seen = pd.to_datetime(latest["timestamp"])
        seconds_ago = (now - last_seen).total_seconds()
        is_online = seconds_ago < 20
        time_text = format_time_ago(seconds_ago)
        
        # Display ML Alert if exists
        ml_status = latest.get("ml_status", "Healthy")
        ml_alert = latest.get("ml_alert", "None")
        
        if ml_status == "Critical":
            st.error(f"🚨 ML CRITICAL ALERT: {ml_alert}")
        elif ml_status == "Warning":
            st.warning(f"⚠️ ML WARNING: {ml_alert}")
    else:
        is_online = False
        time_text = "Unknown"
        ml_status = "Healthy"
else:
    is_online = False
    latest = None
    time_text = "No Device"
    df = pd.DataFrame()

# 4. Status Banner
status_col, empty_col = st.columns([2, 1])
with status_col:
    if is_online:
        st.success(f"🟢 **ONLINE** (Live Data Streaming)")
    else:
        st.error(f"🔴 **OFFLINE** (Last seen: {time_text} ago)")

# 5. Live Metrics (Current State)
st.subheader("Live Engine Status")
m1, m2, m3, m4 = st.columns(4)
m5, m6, m7, m8 = st.columns(4)
m9, m10, m11, m12 = st.columns(4)

if latest is not None:
    # Row 1
    m1.metric("RPM", int(latest.get("RPM", 0)))
    m2.metric("Speed", f"{int(latest.get('Speed', 0))} km/h")
    m3.metric("Engine Load", f"{float(latest.get('EngineLoad', 0))} %")
    m4.metric("Throttle", f"{float(latest.get('ThrottlePos', 0))} %")
    
    # Row 2
    m5.metric("Coolant Temp", f"{float(latest.get('CoolantTemp', 0))} °C")
    m6.metric("Oil Temp", f"{float(latest.get('OilTemp', 0))} °C")
    m7.metric("Intake Temp", f"{float(latest.get('IntakeTemp', 0))} °C")
    m8.metric("Fuel Level", f"{float(latest.get('FuelLevel', 0))} %")
    
    # Row 3
    m9.metric("Voltage", f"{float(latest.get('Voltage', 0))} V")
    m10.metric("MAP Pressure", f"{float(latest.get('MAP', 0))} kPa")
    m11.metric("STFT / LTFT", f"{float(latest.get('STFT', 0))}% / {float(latest.get('LTFT', 0))}%")
    m12.metric("O2 Voltage", f"{float(latest.get('O2Voltage', 0))} V")
else:
    m1.metric("RPM", 0); m2.metric("Speed", 0); m3.metric("Engine Load", 0); m4.metric("Throttle", 0)
    m5.metric("Coolant Temp", 0); m6.metric("Oil Temp", 0); m7.metric("Intake Temp", 0); m8.metric("Fuel Level", 0)
    m9.metric("Voltage", 0); m10.metric("MAP Pressure", 0); m11.metric("STFT / LTFT", "0% / 0%"); m12.metric("O2 Voltage", 0)

# 6. Graphs
st.subheader("Live Telemetry Graph")
if not df.empty:
    df_plot = add_breaks_for_gaps(df, threshold_seconds=10)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_rpm = px.line(df_plot, x="timestamp", y="RPM", title="RPM")
        fig_rpm.update_traces(connectgaps=False, line_color='#FF4B4B')
        st.plotly_chart(fig_rpm, use_container_width=True)
        
        fig_temp = px.line(df_plot, x="timestamp", y=["CoolantTemp", "OilTemp"], title="Engine Temperatures (°C)")
        fig_temp.update_traces(connectgaps=False)
        st.plotly_chart(fig_temp, use_container_width=True)
        
        fig_fuel = px.line(df_plot, x="timestamp", y=["STFT", "LTFT"], title="Fuel Trims (%)")
        fig_fuel.update_traces(connectgaps=False)
        st.plotly_chart(fig_fuel, use_container_width=True)

    with col2:
        fig_speed = px.line(df_plot, x="timestamp", y="Speed", title="Speed (km/h)")
        fig_speed.update_traces(connectgaps=False, line_color='#00CC96')
        st.plotly_chart(fig_speed, use_container_width=True)
        
        fig_load = px.line(df_plot, x="timestamp", y=["EngineLoad", "ThrottlePos"], title="Load & Throttle (%)")
        fig_load.update_traces(connectgaps=False)
        st.plotly_chart(fig_load, use_container_width=True)
        
        fig_map = px.line(df_plot, x="timestamp", y="MAP", title="Manifold Absolute Pressure (kPa)")
        fig_map.update_traces(connectgaps=False, line_color='#AB63FA')
        st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Waiting for historical data...")

# Auto-refresh
if is_online:
    time.sleep(2)
    st.rerun()
else:
    time.sleep(10)
    st.rerun()