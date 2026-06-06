import os
import requests
from datetime import datetime, timedelta, timezone
from typing import List
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
import uvicorn

# 🟢 FIX: Use a global session to pool connections and prevent socket exhaustion during rapid bulk uploads!
session = requests.Session()

app = FastAPI(title="OBD ML Server")

# Your Firebase Database URL
FIREBASE_DB_URL = "https://arapp-feb0f-default-rtdb.firebaseio.com/"

# A simple model representing the incoming data from Unity
class VehicleData(BaseModel):
    device_id: str
    timestamp: str
    RPM: float
    Speed: int
    CoolantTemp: int
    EngineLoad: float
    IntakeTemp: int = 0
    ThrottlePos: float = 0
    Voltage: float = 0
    MAP: int = 0
    STFT: float = 0
    LTFT: float = 0
    O2Voltage: float = 0
    ml_prediction: str = "Healthy"
    ml_future_status: str = "Healthy"
    ml_future_component: str = "None"
    ml_future_hours: float = 0.0

def process_and_upload(data: VehicleData):
    """
    Background task to run ML predictions and upload to Firebase.
    """
    # ---------------------------------------------------------
    # 🧠 FUTURE ML LOGIC GOES HERE
    # ---------------------------------------------------------
    alert_msg = "None"
    status = "Healthy"
    
    # Very basic threshold examples based on your 9-class architecture:
    if data.Voltage > 0 and data.Voltage < 11.5 and data.RPM == 0:
        alert_msg = "Warning: Weak Battery Detected"
        status = "Warning"
    elif data.CoolantTemp > 95:
        alert_msg = "CRITICAL: Engine Overheating"
        status = "Critical"
    elif data.ThrottlePos > 80 and data.MAP < 30: # Example logic
        alert_msg = "Warning: Possible Clogged Air Filter"
        status = "Warning"
    
    # ---------------------------------------------------------
    # ☁️ FIREBASE UPLOAD
    # ---------------------------------------------------------
    try:
        # 1. Update the "Live" state for the dashboard speedometers
        live_payload = data.dict()
        live_payload["ml_status"] = status
        live_payload["ml_alert"] = alert_msg
        
        live_url = f"{FIREBASE_DB_URL}live/{data.device_id}.json"
        session.put(live_url, json=live_payload, timeout=5)
        
        # 2. Append to "History" for the dashboard graphs
        # 🟢 FIX: Use the actual timestamp from the app so delayed uploads stay in perfectly sorted order!
        try:
            dt = datetime.strptime(data.timestamp, "%Y-%m-%d %H:%M:%S")
            time_key = dt.strftime("%Y%m%d_%H%M%S")
        except:
            time_key = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
        history_url = f"{FIREBASE_DB_URL}history/{data.device_id}/{time_key}.json"
        
        # 🟢 FIX: Save ALL data so the raw table doesn't show 0s!
        history_payload = {
            "timestamp": live_payload["timestamp"],
            "RPM": data.RPM,
            "Speed": data.Speed,
            "CoolantTemp": data.CoolantTemp,
            "EngineLoad": data.EngineLoad,
            "Voltage": data.Voltage,
            "IntakeTemp": data.IntakeTemp,
            "MAF": data.MAF,
            "ThrottlePos": data.ThrottlePos,
            "OilTemp": data.OilTemp,
            "MAP": data.MAP,
            "FuelLevel": data.FuelLevel,
            "STFT": data.STFT,
            "LTFT": data.LTFT,
            "O2Voltage": data.O2Voltage
        }
        session.put(history_url, json=history_payload, timeout=5)
        
        # 3. Trim History
        trim_history(data.device_id)
    except Exception as e:
        print(f"Firebase Upload Error: {e}")

def trim_history(device_id: str):
    """ Deletes old records if the history gets too large """
    try:
        # Fetch only the keys (shallow=true) to save bandwidth
        history_url = f"{FIREBASE_DB_URL}history/{device_id}.json?shallow=true"
        response = session.get(history_url, timeout=5)
        if response.status_code == 200 and response.json():
            keys = sorted(list(response.json().keys()))
            if len(keys) > 2000:
                # Delete the oldest keys
                keys_to_delete = keys[:-2000]
                for key in keys_to_delete:
                    del_url = f"{FIREBASE_DB_URL}history/{device_id}/{key}.json"
                    session.delete(del_url, timeout=5)
    except Exception as e:
        print("Trim error:", e)

def process_bulk_upload(data_list: List[VehicleData], background_tasks: BackgroundTasks = None):
    """ Bulk uploads an entire queue to Firebase in a single blazing fast request """
    if not data_list:
        return
        
    try:
        latest = data_list[-1]
        
        # 🟢 FIX: ALWAYS Update Live Node!
        # Reverting to the exact proven architecture from server1.py
        live_payload = latest.dict()
        prediction = latest.ml_prediction
        
        if "Healthy" in prediction:
            live_payload["ml_status"] = "Healthy"
            live_payload["ml_alert"] = "None"
        else:
            live_payload["ml_status"] = "Warning" if prediction in ["Clogged_Filter", "Bad_Alternator"] else "Critical"
            live_payload["ml_alert"] = f"ML DETECTION: {prediction.replace('_', ' ')}"
            
        try:
            dt = datetime.strptime(latest.timestamp, "%Y-%m-%d %H:%M:%S")
            # Convert packet to UTC (Assuming UTC+5 based on system context)
            packet_utc = dt - timedelta(hours=5)
            age_seconds = (datetime.now(timezone.utc).replace(tzinfo=None) - packet_utc).total_seconds()
        except:
            age_seconds = 0
            
        # If packet is newer than 1 hour, it's from the current drive cycle. Inject perfect server time.
        if -3600 < age_seconds < 3600:
            live_payload["server_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            
        session.put(f"{FIREBASE_DB_URL}live/{latest.device_id}.json", json=live_payload, timeout=5).raise_for_status()
        
        # 2. Update History in Bulk (ALL packets at once using PATCH)
        history_updates = {}
        for d in data_list:
            try:
                dt = datetime.strptime(d.timestamp, "%Y-%m-%d %H:%M:%S")
                time_key = dt.strftime("%Y%m%d_%H%M%S")
            except:
                time_key = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Ensure no missing fields
            payload = d.dict()
            history_updates[time_key] = payload
            
        patch_url = f"{FIREBASE_DB_URL}history/{latest.device_id}.json"
        
        # 🟢 FIX: CLOSED LOOP CONFIRMATION
        # raise_for_status() will instantly crash this function if Firebase fails.
        # This prevents Unity from deleting its cache if Firebase didn't actually save the data!
        session.patch(patch_url, json=history_updates, timeout=15).raise_for_status()
        
        # 3. Trim History (IN BACKGROUND)
        # 🟢 FIX: Trimming history requires a GET request that takes 500ms! 
        # Doing this synchronously was slowing down the Unity response loop and causing 8-9 second delays.
        if background_tasks:
            background_tasks.add_task(trim_history, latest.device_id)
        else:
            trim_history(latest.device_id)
        
    except Exception as e:
        print(f"Bulk Upload Error: {e}")
        # Re-raise the exception so FastAPI returns a 500 Server Error to Unity, 
        # forcing Unity to keep the data in its offline cache for a retry!
        raise e

@app.post("/api/upload")
def upload_data(data: List[VehicleData], background_tasks: BackgroundTasks):
    """
    Unity sends data here as a JSON array (bulk upload).
    🟢 FIX: We wait for Firebase to successfully save the data BEFORE returning 200 OK.
    However, we offload the heavy 'trim_history' to a background task so Unity gets an instant response!
    """
    process_bulk_upload(data, background_tasks)
    return {"status": "success", "message": f"Bulk processing {len(data)} items"}

@app.get("/")
def health_check():
    """ Render.com needs this to know the server is alive """
    return {"status": "online", "message": "OBD FastAPI Server is Running!"}

if __name__ == "__main__":
    # Render.com provides the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)