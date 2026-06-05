import time
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. PAGE CONFIG & THEME SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="AR Diagnostic Dashboard", layout="wide", page_icon="ARVIS2.png", initial_sidebar_state="expanded")

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

def format_offline_duration(seconds):
    if seconds < 0: seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    mo, d = divmod(d, 30)
    
    parts = []
    if mo > 0: parts.append(f"{mo} month{'s' if mo != 1 else ''}")
    if d > 0: parts.append(f"{d} day{'s' if d != 1 else ''}")
    if h > 0: parts.append(f"{h} hr{'s' if h != 1 else ''}")
    if m > 0: parts.append(f"{m} min{'s' if m != 1 else ''}")
    if s > 0 or len(parts) == 0: parts.append(f"{s} sec")
    return ", ".join(parts)

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

@st.cache_data(ttl=15)
def get_recent_history_data(device_id):
    try:
        # Fetch only the last 50 records (approx 1.5 minutes) for the incremental cache update!
        # Payload size is practically zero, making it infinitely fast.
        res = requests.get(f"{FIREBASE_DB_URL}history/{device_id}.json?orderBy=\"$key\"&limitToLast=50")
        if res.status_code == 200 and res.json():
            records = list(res.json().values())
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_full_history_data(device_id):
    """ Only called ONCE when the dashboard first loads to build the initial 3-hour cache """
    try:
        res = requests.get(f"{FIREBASE_DB_URL}history/{device_id}.json?orderBy=\"$key\"&limitToLast=6000")
        if res.status_code == 200 and res.json():
            records = list(res.json().values())
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except: pass
    return pd.DataFrame()

# ---------------------------------------------------------
# 3. SIDEBAR (FILLED WITH CONTEXT)
# ---------------------------------------------------------
with st.sidebar:
    # 🟢 FIX: Use columns to perfectly center the new transparent logo and make it larger!
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.image("ARVIS2.png", use_column_width=True)
        
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
st.title("🚗 ARVIS Dashboard")

if device_id:
    latest_raw = get_live_data(device_id)
    
    if latest_raw:
        latest = latest_raw
        
        # 🟢 FIX: Flawless Offline Check with Fallback
        try:
            if "server_timestamp_utc" in latest:
                server_arr_time = pd.to_datetime(latest["server_timestamp_utc"]).tz_localize(None)
                seconds_ago = (datetime.utcnow() - server_arr_time).total_seconds()
            else:
                # Fallback if server.py isn't updated: track local arrival time
                current_packet_time = latest.get("timestamp", "")
                if "fallback_last_packet" not in st.session_state or current_packet_time != st.session_state.get("fallback_last_packet"):
                    st.session_state["fallback_last_packet"] = current_packet_time
                    st.session_state["fallback_arrival_time"] = time.time()
                seconds_ago = time.time() - st.session_state.get("fallback_arrival_time", time.time())
                
            is_online = seconds_ago <= 15
            is_live_data_fresh = seconds_ago <= 4
        except:
            is_online = False
            is_live_data_fresh = False
            seconds_ago = 9999
            
    # 🟢 CRITICAL PERFORMANCE OPTIMIZATION
    # Do NOT download and concatenate the heavy history DataFrame every 1.5 seconds!
    # We only update history every 10 seconds to drastically reduce RAM and CPU load.
    if "last_history_update" not in st.session_state:
        st.session_state.last_history_update = 0
        
    if time.time() - st.session_state.last_history_update > 10.0:
        st.session_state.last_history_update = time.time()
        
        recent_df = get_recent_history_data(device_id).copy()
        if "full_history_df" not in st.session_state:
            st.session_state.full_history_df = get_full_history_data(device_id).copy()
            
        if not recent_df.empty:
            combined = pd.concat([st.session_state.full_history_df, recent_df])
            combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            # Keep only last 2 hours to keep it even lighter
            two_hours_ago = combined['timestamp'].max() - timedelta(hours=2)
            st.session_state.full_history_df = combined[combined['timestamp'] >= two_hours_ago]
            
    df = st.session_state.get("full_history_df", pd.DataFrame())
    
    # Ensure ALL columns exist to prevent crashes
    expected_cols = ["RPM", "Speed", "CoolantTemp", "EngineLoad", "Voltage", 
                     "IntakeTemp", "MAF", "ThrottlePos", "OilTemp", "MAP", 
                     "FuelLevel", "STFT", "LTFT", "O2Voltage"]
    if not df.empty:
        for col in expected_cols:
            if col not in df.columns: df[col] = 0.0
            
    if not latest_raw:
        is_online = False
        is_live_data_fresh = False
        latest = None
        seconds_ago = 9999
else:
    is_online = False
    is_live_data_fresh = False
    latest = None
    df = pd.DataFrame()
    seconds_ago = 9999

# Status Banner
if is_online:
    st.success("🟢 **SYSTEM ONLINE** — Live Data Streaming Active")
else:
    if latest:
        offline_text = format_offline_duration(seconds_ago)
        st.error(f"🔴 **SYSTEM OFFLINE** — Connection lost for {offline_text}")
    else:
        st.error("🔴 **SYSTEM OFFLINE** — No vehicle connected.")

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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Metrics", "📈 Graphs", "📝 Raw Historical Data", "🚨 Alerts"])

# ================= TAB 1: LIVE METRICS =================
with tab1:
    st.subheader("Real-Time Engine Status")
    
    if latest and is_online and is_live_data_fresh:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RPM", int(latest.get("RPM", 0)))
        c2.metric("Speed", f"{int(latest.get('Speed', 0))} km/h")
        c3.metric("Engine Load", f"{float(latest.get('EngineLoad', 0))} %")
        c4.metric("Throttle", f"{float(latest.get('ThrottlePos', 0))} %")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Coolant Temp", f"{float(latest.get('CoolantTemp', 0))} °C")
        c6.metric("Oil Temp", f"{float(latest.get('OilTemp', 0))} °C")
        c7.metric("Intake Temp", f"{float(latest.get('IntakeTemp', 0))} °C")
        c8.metric("Voltage", f"{float(latest.get('Voltage', 0))} V")
        
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("MAP Pressure", f"{float(latest.get('MAP', 0))} kPa")
        c10.metric("MAF Airflow", f"{float(latest.get('MAF', 0))} g/s")
        c11.metric("STFT / LTFT", f"{float(latest.get('STFT', 0))}% / {float(latest.get('LTFT', 0))}%")
        c12.metric("O2 Sensor", f"{float(latest.get('O2Voltage', 0))} V")
    else:
        # Show stale indicators when the feed pauses > 4 seconds
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RPM", "--"); c2.metric("Speed", "-- km/h"); c3.metric("Engine Load", "-- %"); c4.metric("Throttle", "-- %")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Coolant Temp", "-- °C"); c6.metric("Oil Temp", "-- °C"); c7.metric("Intake Temp", "-- °C"); c8.metric("Voltage", "-- V")
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("MAP Pressure", "-- kPa"); c10.metric("MAF Airflow", "-- g/s"); c11.metric("STFT / LTFT", "--% / --%"); c12.metric("O2 Sensor", "-- V")

# ================= TAB 2: GRAPHS (LAST 5 MINS) =================
with tab2:
    if not df.empty:
        # STRICT 5-MINUTE WINDOW CUTOFF
        five_mins_ago = df["timestamp"].max() - timedelta(minutes=5)
        df_graphs = df[df["timestamp"] >= five_mins_ago].copy()
        
        df_plot = add_breaks_for_gaps(df_graphs, threshold_seconds=5)

        # 🟢 FIX: Drastic Performance Optimization
        # We completely removed Plotly (which is extremely heavy for the server) 
        # and replaced it with Streamlit's native Altair line_charts.
        # This shifts the rendering load to the browser, making it run lightning-fast!
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("###### Engine RPM")
            st.line_chart(df_plot, x="timestamp", y="RPM", color="#FF4B4B", height=200, use_container_width=True)
            st.markdown("###### Coolant Temp (°C)")
            st.line_chart(df_plot, x="timestamp", y="CoolantTemp", color="#FFA500", height=200, use_container_width=True)
            st.markdown("###### MAP Pressure (kPa)")
            st.line_chart(df_plot, x="timestamp", y="MAP", color="#AB63FA", height=200, use_container_width=True)
            st.markdown("###### Short Term Fuel Trim (%)")
            st.line_chart(df_plot, x="timestamp", y="STFT", color="#E2D9F3", height=200, use_container_width=True)
            
        with g2:
            st.markdown("###### Vehicle Speed (km/h)")
            st.line_chart(df_plot, x="timestamp", y="Speed", color="#00CC96", height=200, use_container_width=True)
            st.markdown("###### Oil Temp (°C)")
            st.line_chart(df_plot, x="timestamp", y="OilTemp", color="#F4D03F", height=200, use_container_width=True)
            st.markdown("###### Intake Temp (°C)")
            st.line_chart(df_plot, x="timestamp", y="IntakeTemp", color="#58D68D", height=200, use_container_width=True)
            st.markdown("###### Long Term Fuel Trim (%)")
            st.line_chart(df_plot, x="timestamp", y="LTFT", color="#A569BD", height=200, use_container_width=True)
            
        with g3:
            st.markdown("###### Engine Load (%)")
            st.line_chart(df_plot, x="timestamp", y="EngineLoad", color="#636EFA", height=200, use_container_width=True)
            st.markdown("###### Throttle Position (%)")
            st.line_chart(df_plot, x="timestamp", y="ThrottlePos", color="#1ABC9C", height=200, use_container_width=True)
            st.markdown("###### Battery Voltage (V)")
            st.line_chart(df_plot, x="timestamp", y="Voltage", color="#F39C12", height=200, use_container_width=True)
            st.markdown("###### O2 Sensor (V)")
            st.line_chart(df_plot, x="timestamp", y="O2Voltage", color="#E74C3C", height=200, use_container_width=True)
            
    else:
        st.info("No historical data available yet. Start the engine to generate graphs!")

# ================= TAB 3: TABULAR DATA =================
with tab3:
    st.subheader("Historical Telemetry Log")
    
    if not df.empty:
        df_table = df.copy()
        
        # Clean up the format so it's not messy!
        # The timestamp is already in Local Time from the phone
        df_table['Date'] = df_table['timestamp'].dt.strftime('%Y-%m-%d')
        df_table['Time (Local)'] = df_table['timestamp'].dt.strftime('%H:%M:%S')
        
        # Reorder columns to put Date and Time first, drop the raw timestamp
        cols = ['Date', 'Time (Local)'] + [c for c in df_table.columns if c not in ['Date', 'Time (Local)', 'timestamp']]
        df_table = df_table[cols]
        
        st.caption("Displaying the full 3-hour history seamlessly from the local memory cache.")
            
        # Display perfectly sorted, most recent first, without the ugly index column
        st.dataframe(df_table.sort_values(["Date", "Time (Local)"], ascending=[False, False]), hide_index=True, use_container_width=True)
    else:
        st.info("Database is entirely blank. No historical logs exist.")

# ================= TAB 4: ALERTS =================
with tab4:
    st.subheader("Historical ML Alerts (Last 7 Days)")
    st.markdown("Automated AI Diagnostic engine scanning telemetry history to isolate confirmed component failures.")
    
    if not df.empty and "ml_prediction" in df.columns:
        try:
            df_alerts = df.copy()
            # 🟢 FIX: Calculate Block IDs *BEFORE* filtering!
            # This ensures that if Misfire happens, then Healthy, then Misfire again,
            # they are treated as completely separate alerts, rather than merged together!
            df_alerts['Block'] = (df_alerts['ml_prediction'] != df_alerts['ml_prediction'].shift(1)).cumsum()
            
            # Now filter out healthy states
            df_faults = df_alerts[~df_alerts['ml_prediction'].str.contains("Healthy", na=False)].copy()
            
            if df_faults.empty:
                st.success("✅ **No confirmed alerts in the recent history.** Your engine is running perfectly!")
            else:
                # Group by these contiguous blocks
                confirmed_alerts = []
                for block_id, group in df_faults.groupby('Block'):
                    if len(group) >= 3: # MUST PERSIST for at least 3 packets to avoid false edge alarms
                        start_time = group['timestamp'].iloc[0]
                        end_time = group['timestamp'].iloc[-1]
                        alert_type = group['ml_prediction'].iloc[0].replace("_", " ")
                        
                        # 🟢 FIX: Exact Timestamp Math
                        t_start = pd.to_datetime(start_time)
                        t_end = pd.to_datetime(end_time)
                        exact_seconds = (t_end - t_start).total_seconds()
                        
                        # Avoid 0 seconds if the packet gap is extremely small
                        if exact_seconds < 1: exact_seconds = len(group) * 1.5
                        
                        confirmed_alerts.append({
                            "Start": start_time,
                            "End": end_time,
                            "Alert": alert_type,
                            "DurationSeconds": exact_seconds
                        })
                
                # Reverse list to show newest first
                confirmed_alerts.reverse()
                
                if len(confirmed_alerts) == 0:
                    st.success("✅ **No confirmed alerts.** (Some minor sensor edges were detected but discarded as noise).")
                else:
                    for alert in confirmed_alerts:
                        # Draw beautiful UI Banners for each confirmed alert
                        bg_color = "#4a0f0f" if alert["Alert"] in ["Misfire", "Overheat"] else "#4a3c0f"
                        icon = "🔥" if alert["Alert"] == "Overheating" else "⚡" if alert["Alert"] == "Bad Alternator" else "🚨"
                        
                        # 🟢 FIX: Format into Hours, Minutes, Seconds
                        duration_text = format_offline_duration(alert['DurationSeconds'])
                        
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #ff4b4b;">
                            <h4 style="margin: 0; color: white;">{icon} CONFIRMED: {alert['Alert']}</h4>
                            <p style="margin: 5px 0 0 0; color: #d1d1d1; font-size: 14px;">
                                <b>Component Affected:</b> Engine / Diagnostics<br>
                                <b>Time:</b> {alert['Start']} to {alert['End']}<br>
                                <b>Sustained Duration:</b> {duration_text}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Error processing alerts: {str(e)}")
    else:
        st.info("Waiting for data to run diagnostics...")

# ---------------------------------------------------------
# 7. AUTO-REFRESH LOGIC
# ---------------------------------------------------------
# 🟢 FIX: Optimized Refresh Rates
# Refreshing too fast blocks the browser and creates lag/stuttering. 
# Unity only uploads every 2 seconds anyway!
if is_online:
    time.sleep(1.5) # Nyquist offset: slightly out of sync with Unity's 1.8s to avoid harmonic delay
    st.rerun()
else:
    time.sleep(5) # Slow down when offline to completely unblock the server
    st.rerun()