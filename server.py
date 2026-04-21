import os
import requests
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
import uvicorn

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
    MAF: float = 0
    ThrottlePos: float = 0
    Voltage: float = 0
    OilTemp: int = 0
    MAP: int = 0
    FuelLevel: float = 0
    STFT: float = 0
    LTFT: float = 0
    O2Voltage: float = 0

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
    # 1. Update the "Live" state for the dashboard speedometers
    live_payload = data.dict()
    live_payload["ml_status"] = status
    live_payload["ml_alert"] = alert_msg
    
    live_url = f"{FIREBASE_DB_URL}live/{data.device_id}.json"
    requests.put(live_url, json=live_payload)
    
    # 2. Append to "History" for the dashboard graphs
    # We use a timestamp key so Firebase automatically orders it
    time_key = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_url = f"{FIREBASE_DB_URL}history/{data.device_id}/{time_key}.json"
    
    # Only save the crucial data for graphs to save Firebase storage
    history_payload = {
        "timestamp": live_payload["timestamp"],
        "RPM": data.RPM,
        "Speed": data.Speed,
        "CoolantTemp": data.CoolantTemp,
        "EngineLoad": data.EngineLoad,
        "Voltage": data.Voltage
    }
    requests.put(history_url, json=history_payload)
    
    # 3. Trim History (Keep only the last 2000 records to prevent Firebase from filling up)
    trim_history(data.device_id)

def trim_history(device_id: str):
    """ Deletes old records if the history gets too large """
    try:
        # Fetch only the keys (shallow=true) to save bandwidth
        history_url = f"{FIREBASE_DB_URL}history/{device_id}.json?shallow=true"
        response = requests.get(history_url)
        if response.status_code == 200 and response.json():
            keys = sorted(list(response.json().keys()))
            if len(keys) > 2000:
                # Delete the oldest keys
                keys_to_delete = keys[:-2000]
                for key in keys_to_delete:
                    del_url = f"{FIREBASE_DB_URL}history/{device_id}/{key}.json"
                    requests.delete(del_url)
    except Exception as e:
        print("Trim error:", e)

@app.post("/api/upload")
async def upload_data(data: VehicleData, background_tasks: BackgroundTasks):
    """
    Unity sends data here. We immediately return 200 OK so Unity doesn't freeze,
    then we process the ML and Firebase upload in the background.
    """
    background_tasks.add_task(process_and_upload, data)
    return {"status": "success", "message": "Data received and processing"}

@app.get("/")
def health_check():
    """ Render.com needs this to know the server is alive """
    return {"status": "online", "message": "OBD FastAPI Server is Running!"}

if __name__ == "__main__":
    # Render.com provides the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)