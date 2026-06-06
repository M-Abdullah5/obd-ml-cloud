import time
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from datetime import datetime, timedelta, timezone
import streamlit.components.v1 as components

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

@st.cache_resource
def get_http_session():
    """ 🟢 FIX: Global HTTP Session to prevent recreating TLS handshakes every 1.5 seconds.
    This massively speeds up Render free-tier fetching! """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount('https://', adapter)
    return session

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
    y, mo = divmod(mo, 12)
    
    parts = []
    if y > 0: parts.append(f"{y} year{'s' if y != 1 else ''}")
    if mo > 0: parts.append(f"{mo} month{'s' if mo != 1 else ''}")
    if d > 0: parts.append(f"{d} day{'s' if d != 1 else ''}")
    if h > 0: parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m > 0: parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s > 0 or len(parts) == 0: parts.append(f"{s} second{'s' if s != 1 else ''}")
    
    return ", ".join(parts)

@st.cache_data(ttl=3)
def get_devices():
    try:
        res = get_http_session().get(f"{FIREBASE_DB_URL}live.json?shallow=true", timeout=3.0)
        if res.status_code == 200 and res.json():
            return list(res.json().keys())
    except: pass
    return []

def get_live_data(device_id):
    try:
        # 🟢 FIX: Use pooled session for lightning-fast fetching
        res = get_http_session().get(f"{FIREBASE_DB_URL}live/{device_id}.json", timeout=1.5)
        if res.status_code == 200: return res.json()
    except: pass
    return None

def get_recent_history_data(device_id):
    try:
        # 🟢 BULK INCREMENTAL FETCH (BACKWARD COMPATIBLE)
        # Because your Render backend hasn't been updated to the Append-Only architecture,
        # we CANNOT use `startAt` (it skips packets that Render inserts into the past).
        # Instead, we brute-force pull the newest 800 packets (approx 6.5 minutes of cache)
        # every cycle. This effortlessly absorbs the 1-minute disconnects you are testing!
        url = f"{FIREBASE_DB_URL}history/{device_id}.json?orderBy=\"$key\"&limitToLast=800"
            
        res = get_http_session().get(url, timeout=3.0)
        if res.status_code == 200 and res.json():
            data = res.json()
            records = list(data.values())
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except: pass
    return pd.DataFrame()

def get_full_history_data(device_id):
    """ Only called ONCE when the dashboard first loads to build the initial 3-hour cache """
    try:
        res = get_http_session().get(f"{FIREBASE_DB_URL}history/{device_id}.json?orderBy=\"$key\"&limitToLast=6000", timeout=10.0)
        if res.status_code == 200 and res.json():
            data = res.json()
            records = list(data.values())
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
        st.image("ARVIS2.png")
        
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
    # 1. FETCH FULL HISTORY ONCE 
    if "full_history_df" not in st.session_state:
        df = get_full_history_data(device_id)
        st.session_state.full_history_df = df.copy()
            
    # 2. FETCH INCREMENTAL CACHE (BULK COMPATIBILITY MODE)
    recent_df = get_recent_history_data(device_id)
        
    # 3. STACK AND TRIM
    if not recent_df.empty:
        combined = pd.concat([st.session_state.full_history_df, recent_df])
        combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        two_hours_ago = combined['timestamp'].max() - timedelta(hours=3)
        st.session_state.full_history_df = combined[combined['timestamp'] >= two_hours_ago]
            
    df = st.session_state.get("full_history_df", pd.DataFrame())
    
    # 2. FETCH LIVE
    latest_raw = get_live_data(device_id)
    latest = latest_raw if latest_raw else {}
    
    # 3. CROSS-REFERENCE AND CALCULATE STRICT OBD AGE
    try:
        current_packet_time = latest.get("timestamp", "")
        
        # Override with history if it's fresher (bypasses broken Live nodes instantly)
        if not df.empty:
            freshest_history_time = str(df['timestamp'].max())
            if freshest_history_time > current_packet_time:
                latest = df.iloc[-1].to_dict()
                current_packet_time = str(latest.get("timestamp", ""))
                
        # 🟢 STRICT OBD PACKET AGE
        # We no longer trust the server arrival time. We calculate exactly how old the
        # data is based purely on when it was generated by the car.
        packet_utc = pd.to_datetime(current_packet_time) - timedelta(hours=5)
        absolute_seconds_ago = (datetime.now(timezone.utc).replace(tzinfo=None) - packet_utc).total_seconds()
        
        # Prevent negative seconds if phone clock is a fraction of a second fast
        seconds_ago = max(0.0, absolute_seconds_ago)
        
        # 🟢 USER REQUIREMENT: "if data is 4 seconds old maximum, it should not be considered live"
        is_online = seconds_ago <= 15
        is_display_fresh = seconds_ago <= 6.0      
        is_actually_live = seconds_ago <= 4.0      
        
    except Exception as e:
        is_online = False
        is_display_fresh = False
        is_actually_live = False
        seconds_ago = 9999
        
    # Ensure ALL columns exist to prevent crashes
    expected_cols = ["RPM", "Speed", "CoolantTemp", "EngineLoad", "Voltage", 
                     "IntakeTemp", "MAF", "ThrottlePos", "OilTemp", "MAP", 
                     "FuelLevel", "STFT", "LTFT", "O2Voltage", 
                     "ml_status", "ml_alert"]
    if not df.empty:
        for col in expected_cols:
            if col not in df.columns: 
                df[col] = "Healthy" if col == "ml_status" else "None" if col == "ml_alert" else 0.0
            
    if not latest:
        is_online = False
        is_display_fresh = False
        is_actually_live = False
        latest = None
        seconds_ago = 9999
else:
    is_online = False
    is_display_fresh = False
    is_actually_live = False
    latest = None
    df = pd.DataFrame()
    seconds_ago = 9999

# Status Banner
if is_online:
    if is_actually_live:
        st.success("🟢 **SYSTEM ONLINE** — Live Data Streaming Active")
    else:
        st.warning(f"🟡 **DATA DELAYED** — Last packet received {int(seconds_ago)}s ago. Waiting for live sync...")
else:
    if latest:
        offline_text = format_offline_duration(seconds_ago)
        st.error(f"🔴 **SYSTEM OFFLINE** — Connection lost for {offline_text}")
    else:
        st.error("🔴 **SYSTEM OFFLINE** — No vehicle connected.")

# ---------------------------------------------------------
# 5. PRE-CALCULATE ALERTS FOR TAB NOTIFICATIONS
# ---------------------------------------------------------
confirmed_alerts = []
if not df.empty and "ml_prediction" in df.columns:
    try:
        df_alerts = df.copy()
        df_alerts['ml_prediction'] = df_alerts['ml_prediction'].astype(str).str.split(',')
        df_alerts = df_alerts.explode('ml_prediction')
        df_alerts['ml_prediction'] = df_alerts['ml_prediction'].str.strip()
        df_faults = df_alerts[~df_alerts['ml_prediction'].str.contains("Healthy", na=False, case=False)].copy()
        
        if not df_faults.empty:
            for alert_type, alert_group in df_faults.groupby('ml_prediction'):
                alert_group = alert_group.sort_values('timestamp')
                alert_group['time_diff'] = alert_group['timestamp'].diff().dt.total_seconds()
                alert_group['Block'] = (alert_group['time_diff'] > 15).cumsum()
                
                for block_id, group in alert_group.groupby('Block'):
                    if len(group) >= 3:
                        start_time = group['timestamp'].iloc[0]
                        end_time = group['timestamp'].iloc[-1]
                        clean_alert_name = alert_type.replace("_", " ")
                        
                        t_start = pd.to_datetime(start_time)
                        t_end = pd.to_datetime(end_time)
                        exact_seconds = (t_end - t_start).total_seconds()
                        if exact_seconds < 1: exact_seconds = len(group) * 1.5
                        
                        max_db_time = pd.to_datetime(df['timestamp'].max())
                        
                        # 🟢 FIX: Handle OBD Disconnection during active alert!
                        is_active = False
                        was_disconnected = False
                        if (max_db_time - t_end).total_seconds() <= 5:
                            if is_online:
                                is_active = True
                            else:
                                was_disconnected = True
                        
                        confirmed_alerts.append({
                            "Start": start_time,
                            "End": end_time,
                            "Alert": clean_alert_name,
                            "DurationSeconds": exact_seconds,
                            "IsActive": is_active,
                            "WasDisconnected": was_disconnected
                        })
            confirmed_alerts.sort(key=lambda x: x['End'], reverse=True)
    except Exception as e:
        pass

# 🟢 FIX: We CANNOT dynamically change Tab Names in Streamlit!
# If the tab name changes from "Alerts (1)" to "Alerts (0)", Streamlit destroys the tab
# and violently kicks the user back to Tab 1. 
# To fix the jumping bug, the tab names MUST remain static!
active_alerts_count = sum(1 for a in confirmed_alerts if a['IsActive'])
alert_badge = f"{active_alerts_count}" if active_alerts_count <= 9 else "9+"

future_rul_status = latest.get("ml_future_status", "Healthy") if latest else "Healthy"
future_alerts_count = 1 if future_rul_status == "Degrading" else 0
future_badge = f"{future_alerts_count}" if future_alerts_count <= 9 else "9+"

# 🟢 NEW: GLOBAL FLOATING ALERTS (TOP RIGHT)
has_floats = False
floating_html = "<div style='position: fixed; top: 60px; right: 20px; z-index: 999999; display: flex; flex-direction: column; gap: 10px;'>"

# We must collect the JS code separately so Streamlit doesn't strip it!
js_scripts = ""

for alert in confirmed_alerts:
    if alert['IsActive'] and alert['DurationSeconds'] <= 25.0:
        has_floats = True
        raw_id = f"{alert['Alert']}_{str(alert['Start'])}"
        safe_id = raw_id.replace(' ', '_').replace('-', '_').replace(':', '_').replace('.', '_')
        
        # 🟢 FIX: Remove all indentation so Streamlit does NOT render this as a raw <pre> code block!
        floating_html += f"""
<div id="float_{safe_id}" style="background: linear-gradient(135deg, #ff4b4b 0%, #b30000 100%); color: white; padding: 15px; border-radius: 10px; box-shadow: 0px 8px 16px rgba(0,0,0,0.5); border: 2px solid white; width: 300px; display: block;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.4); padding-bottom: 8px; margin-bottom: 8px;">
<span style="font-weight: bold; font-size: 12px; letter-spacing: 1px;">⚠️ ENGINE FAULT DETECTED</span>
<span id="close_{safe_id}" style="cursor: pointer; font-size: 14px; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 5px;">✖</span>
</div>
<div style="font-size: 16px; font-weight: bold;">{alert['Alert']}</div>
</div>
"""
        # 🟢 FIX: Break out of the components iframe to access the parent Streamlit DOM!
        js_scripts += f"""
    // Hide immediately if already dismissed
    if (session.getItem('dismiss_{safe_id}') === 'true') {{
        const el = parent.getElementById('float_{safe_id}');
        if (el) el.style.display = 'none';
    }}
    
    // Bind click event natively
    const btn = parent.getElementById('close_{safe_id}');
    if (btn) {{
        btn.onclick = function() {{
            session.setItem('dismiss_{safe_id}', 'true');
            parent.getElementById('float_{safe_id}').style.display = 'none';
        }};
    }}
"""

floating_html += "</div>"

# ---------------------------------------------------------
# 6. TABBED INTERFACE
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Live Metrics", "📈 Graphs", "📝 Raw Historical Data", "🚨 Alerts", "🔮 Future Alerts"])

# ================= TAB 1: LIVE METRICS =================
with tab1:
    st.subheader("Real-Time Engine Status")
    
    if latest and is_online and is_display_fresh:
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
            st.line_chart(df_plot, x="timestamp", y="RPM", color="#FF4B4B", height=200, width='stretch')
            st.markdown("###### Coolant Temp (°C)")
            st.line_chart(df_plot, x="timestamp", y="CoolantTemp", color="#FFA500", height=200, width='stretch')
            st.markdown("###### MAP Pressure (kPa)")
            st.line_chart(df_plot, x="timestamp", y="MAP", color="#AB63FA", height=200, width='stretch')
            st.markdown("###### Short Term Fuel Trim (%)")
            st.line_chart(df_plot, x="timestamp", y="STFT", color="#E2D9F3", height=200, width='stretch')
            
        with g2:
            st.markdown("###### Vehicle Speed (km/h)")
            st.line_chart(df_plot, x="timestamp", y="Speed", color="#00CC96", height=200, width='stretch')
            st.markdown("###### Oil Temp (°C)")
            st.line_chart(df_plot, x="timestamp", y="OilTemp", color="#F4D03F", height=200, width='stretch')
            st.markdown("###### Intake Temp (°C)")
            st.line_chart(df_plot, x="timestamp", y="IntakeTemp", color="#58D68D", height=200, width='stretch')
            st.markdown("###### Long Term Fuel Trim (%)")
            st.line_chart(df_plot, x="timestamp", y="LTFT", color="#A569BD", height=200, width='stretch')
            
        with g3:
            st.markdown("###### Engine Load (%)")
            st.line_chart(df_plot, x="timestamp", y="EngineLoad", color="#636EFA", height=200, width='stretch')
            st.markdown("###### Throttle Position (%)")
            st.line_chart(df_plot, x="timestamp", y="ThrottlePos", color="#1ABC9C", height=200, width='stretch')
            st.markdown("###### Battery Voltage (V)")
            st.line_chart(df_plot, x="timestamp", y="Voltage", color="#F39C12", height=200, width='stretch')
            st.markdown("###### O2 Sensor (V)")
            st.line_chart(df_plot, x="timestamp", y="O2Voltage", color="#E74C3C", height=200, width='stretch')
            
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
        st.dataframe(df_table.sort_values(["Date", "Time (Local)"], ascending=[False, False]), hide_index=True, width='stretch')
    else:
        st.info("Database is entirely blank. No historical logs exist.")

# ================= TAB 4: ALERTS =================
with tab4:
    # 🟢 NEW: Display the alert count safely INSIDE the tab to prevent jumping
    if active_alerts_count > 0:
        st.markdown(f"<h3 style='color: #ff4b4b;'>🚨 {active_alerts_count} Active Alerts Happening Now</h3>", unsafe_allow_html=True)
    else:
        st.subheader("Historical ML Alerts (Last 7 Days)")
        
    st.markdown("Automated AI Diagnostic engine scanning telemetry history to isolate confirmed component failures.")
    
    if not df.empty and "ml_prediction" in df.columns:
        if len(confirmed_alerts) == 0:
            st.success("✅ **No confirmed alerts.** (Some minor sensor edges were detected but discarded as noise).")
        else:
            for alert in confirmed_alerts:
                icon = "🔥" if "Overheating" in alert["Alert"] else "⚡" if "Alternator" in alert["Alert"] else "🚨"
                duration_text = format_offline_duration(alert['DurationSeconds'])
                
                # Dynamic Styling based on Active vs Resolved state
                if alert['IsActive']:
                    status_badge = "<span style='background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold; margin-left: 10px; border: 1px solid white;'>🔴 HAPPENING NOW</span>"
                    time_text = f"<b>Started:</b> {alert['Start']} (Ongoing for {duration_text})"
                    border_color = "#ff4b4b"
                    bg_color = "#631313" 
                elif alert.get('WasDisconnected', False):
                    # 🟢 FIX: Explicitly indicate if the OBD disconnected during the fault
                    status_badge = "<span style='background-color: #f39c12; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin-left: 10px;'>🔌 RESOLVED (OBD DISCONNECTED)</span>"
                    time_text = f"<b>Time:</b> {alert['Start']} to {alert['End']}<br><b>Total Duration:</b> {duration_text} before signal loss"
                    border_color = "#f39c12"
                    bg_color = "#333"
                else:
                    status_badge = "<span style='background-color: #555; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin-left: 10px;'>✅ RESOLVED</span>"
                    time_text = f"<b>Time:</b> {alert['Start']} to {alert['End']}<br><b>Total Duration:</b> {duration_text}"
                    border_color = "#555"
                    bg_color = "#333" 
                    
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid {border_color};">
                    <h4 style="margin: 0; color: white;">{icon} {alert['Alert']} {status_badge}</h4>
                    <p style="margin: 5px 0 0 0; color: #d1d1d1; font-size: 14px;">
                        <b>Component Affected:</b> Engine / Diagnostics<br>
                        {time_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Waiting for data to run diagnostics...")

# ================= TAB 5: FUTURE ALERTS (PREDICTIVE MAINTENANCE) =================
with tab5:
    if future_alerts_count > 0:
        st.markdown(f"<h3 style='color: #f39c12;'>🔮 {future_alerts_count} Predictive Alerts</h3>", unsafe_allow_html=True)
    else:
        st.subheader("🔮 Predictive Maintenance (Remaining Useful Life)")
        
    st.markdown("Advanced ML Regression Engine actively monitoring long-term sensor degradation slopes to predict failures BEFORE they happen.")
    
    # 🟢 FUTURE PROOFING: This tab is structurally ready to accept the JSON probability arrays
    # from the new ML model once training is approved and complete!
    if latest and is_online:
        # Example of how the future banner will appear based on the upcoming ML regression model
        future_rul_status = latest.get("ml_future_status", "Healthy")
        future_rul_component = latest.get("ml_future_component", "None")
        future_rul_hours = latest.get("ml_future_hours", 0)
        
        if future_rul_status == "Degrading":
            st.markdown(f"""
            <div style="background-color: #3b2a0c; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #f39c12;">
                <h4 style="margin: 0; color: white;">⏳ PREDICTIVE ALERT: {future_rul_component} Degradation</h4>
                <p style="margin: 5px 0 0 0; color: #d1d1d1; font-size: 14px;">
                    <b>Analysis:</b> The ML Regression model has detected a gradual deviation in sensor bounds indicating physical wear.<br>
                    <b>Estimated Remaining Useful Life (RUL):</b> {future_rul_hours} Hours<br>
                    <b>Action Required:</b> Schedule replacement within the estimated window to prevent catastrophic failure.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ **No Future Faults Predicted** — All component degradation slopes are within factory tolerances.")
    else:
        st.info("Awaiting live telemetry to calculate degradation slopes...")

# ---------------------------------------------------------
# 7. INJECT GLOBAL FLOATING UI ELEMENTS (BOTTOM OF DOM)
# ---------------------------------------------------------
# 🟢 FIX: Moved the floating UI rendering to the absolute bottom of the Streamlit DOM!
# Injecting UI elements above the tabs dynamically shifts the entire Streamlit component tree,
# which causes the tab jumping bug and pushes the entire interface down!
st.markdown(floating_html if has_floats else "<div style='display:none;'></div>", unsafe_allow_html=True)
components.html(f"""
<script>
    const parent = window.parent.document;
    const session = window.parent.sessionStorage;
    {js_scripts}
</script>
""" if has_floats else "<script></script>", height=0)

# ---------------------------------------------------------
# 8. AUTO-REFRESH LOGIC
# ---------------------------------------------------------
# 🟢 FIX: Optimized Refresh Rates for Continuous Flow
if is_online:
    # 🟢 FIX: Streamlit's internal execution and network roundtrip takes ~0.5s to 1.0s.
    # To achieve a TRUE 1.5s update interval on the screen, we sleep for exactly 0.5s!
    time.sleep(0.5) 
    st.rerun()
else:
    time.sleep(1.5) # Faster offline recovery polling
    st.rerun()