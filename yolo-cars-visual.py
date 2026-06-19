import cv2
import time
import os
import re
import threading
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
RTSP_URL = "https://tcnvr6.taichung.gov.tw/4280f46e"
MODEL_PATH = "yolo26x.pt"  
#MODEL_PATH = "yolo11n.pt"  
#MODEL_PATH = "rtdetr-l.pt"  

"""
coco class
2: car, 5: bus, 7: truck
"""
CSV_LOG_PATH = "traffic_sustainability_report_visual.csv"
TEGRASTATS_BIN = "/usr/bin/tegrastats"

CARBON_INTENSITY = 0.509  # Taiwan Carbon Intensity Factor (kg CO2/kWh)
TARGET_CLASSES = [2,5,7]      # COCO Car index

# ==========================================
# REGION OF INTEREST (ROI) CONFIGURATION
# change this points to apply the region masking 
# ==========================================
ROI_POINTS = np.array([    
    [0,  470],  # Point 1: Bottom-Left
    [0,  100],   # Point 2: Top-Left     
    [720,  220],   # Point 3: Top-Right
    [720,  270]   # Point 4: Bottom-Right
], dtype=np.int32)


# ==========================================
# HARDWARE MONITORING CLASS
# ==========================================
class TegrastatsPowerMonitor:
    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.total_energy_j = 0.0
        self._stop = threading.Event()
        self._last_power_w = None
        self._last_ts = None
        self.power_samples = []

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join()

    def reset_interval(self):
        self.total_energy_j = 0.0
        self.power_samples = []
        self._last_power_w = self._last_power_w if hasattr(self, '_last_power_w') else None

    def _run(self):
        proc = subprocess.Popen(
            [TEGRASTATS_BIN, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, text=True
        )
        for line in proc.stdout:
            if self._stop.is_set(): break
            match = re.search(r'VDD_IN (\d+)mW', line)
            if match:
                power_w = int(match.group(1)) / 1000.0
                now = time.time()
                if self._last_power_w is not None:
                    dt = now - self._last_ts
                    self.total_energy_j += ((self._last_power_w + power_w) / 2.0) * dt
                self._last_power_w = power_w
                self._last_ts = now
                self.power_samples.append(power_w)
        proc.terminate()


# ==========================================
# CORE ALGORITHMIC UTILITIES
# ==========================================
def apply_roi_mask(frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [ROI_POINTS], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

def get_density_class(count):
    if count == 0: return "Empty"
    elif 1 <= count <= 2: return "Low"
    elif 3 <= count <= 4: return "Medium"
    elif 5 <= count <= 6: return "High"
    else: return "Congested"

def initialize_csv():
    if not os.path.exists(CSV_LOG_PATH):
        df = pd.DataFrame(columns=[
            "Timestamp", "Day_of_Week", "Time_Interval", 
            "Seconds_Empty", "Seconds_Low", "Seconds_Medium", "Seconds_High", "Seconds_Congested",
            "Avg_Cars_Per_Sec", "Max_Cars_Per_Sec",
            "Avg_Power_Watts", "Max_Power_Watts", "Interval_Energy_kWh",
            "Interval_Carbon_gCO2", "Cumulative_Carbon_kgCO2"
        ])
        df.to_csv(CSV_LOG_PATH, index=False)

def write_to_report(minute_timestamp, metrics, power_monitor, cumulative_carbon_kg):
    dt = datetime.strptime(minute_timestamp, "%Y-%m-%d %H:%M")
    day_name = dt.strftime("%A")
    time_interval = dt.strftime("%H:%M")
    
    avg_cars = round(sum(metrics["counts"]) / len(metrics["counts"]), 2) if metrics["counts"] else 0
    max_cars = max(metrics["counts"]) if metrics["counts"] else 0
    
    samples = power_monitor.power_samples
    avg_power = round(sum(samples) / len(samples), 2) if samples else 0
    max_power = max(samples) if samples else 0
    
    interval_kwh = (power_monitor.total_energy_j / 3600.0) / 1000.0
    interval_gCO2 = interval_kwh * CARBON_INTENSITY * 1000.0

    new_row = {
        "Timestamp": minute_timestamp,
        "Day_of_Week": day_name,
        "Time_Interval": time_interval,
        "Seconds_Empty": metrics["Empty"],
        "Seconds_Low": metrics["Low"],
        "Seconds_Medium": metrics["Medium"],
        "Seconds_High": metrics["High"],
        "Seconds_Congested": metrics["Congested"],
        "Avg_Cars_Per_Sec": avg_cars,
        "Max_Cars_Per_Sec": max_cars,
        "Avg_Power_Watts": avg_power,
        "Max_Power_Watts": max_power,
        "Interval_Energy_kWh": round(interval_kwh, 6),
        "Interval_Carbon_gCO2": round(interval_gCO2, 4),
        "Cumulative_Carbon_kgCO2": round(cumulative_carbon_kg, 6)
    }
    
    df = pd.DataFrame([new_row])
    df.to_csv(CSV_LOG_PATH, mode='a', header=False, index=False)
    print(f"\n--> [LOGGED] {minute_timestamp} | Flow: {avg_cars} cars/sec | Power: {avg_power}W\n")


# ==========================================
# MAIN PIPELINE WITH VISUALIZATION
# ==========================================
def main():
    initialize_csv()
    
    monitor = TegrastatsPowerMonitor(interval_ms=500)
    monitor.start()
    
    model = YOLO(MODEL_PATH)
    
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Error: Could not link to RTSP stream.")
        monitor.stop()
        return

    # Create a resizable visual window container
    cv2.namedWindow("Car Counting Detection - Edge STM Experiment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Car Counting Detection - Edge STM Experiment", 720, 480)

    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    minute_metrics = {"Empty": 0, "Low": 0, "Medium": 0, "High": 0, "Congested": 0, "counts": []}
    
    last_second_time = time.time()
    cumulative_energy_kwh = 0.0
    
    # Store persistent box tracks across display loop cycles
    active_tracks = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(5)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                continue

            now = datetime.now()
            now_minute = now.strftime("%Y-%m-%d %H:%M")
            
            # Minute Rollover Handling
            if now_minute != current_minute:
                this_minute_kwh = (monitor.total_energy_j / 3600.0) / 1000.0
                cumulative_energy_kwh += this_minute_kwh
                cumulative_carbon_kg = cumulative_energy_kwh * CARBON_INTENSITY
                
                if minute_metrics["counts"]:
                    write_to_report(current_minute, minute_metrics, monitor, cumulative_carbon_kg)
                
                current_minute = now_minute
                minute_metrics = {"Empty": 0, "Low": 0, "Medium": 0, "High": 0, "Congested": 0, "counts": []}
                monitor.reset_interval()

            # Process AI tracking exactly once per second
            if time.time() - last_second_time >= 1.0:
                last_second_time = time.time()
                
                masked_frame = apply_roi_mask(frame)
                results = model.track(masked_frame, classes=TARGET_CLASSES, persist=True, verbose=False) #uncomment this line to detect some class
                #results = model.track(masked_frame, persist=True, verbose=False) #uncomment this line to detect all class
                
                active_tracks = []
                car_count = 0
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    car_count = len(results[0].boxes.id)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, ids):
                        active_tracks.append((box, track_id))
                
                density = get_density_class(car_count)
                minute_metrics[density] += 1
                minute_metrics["counts"].append(car_count)
                
                current_w = monitor.power_samples[-1] if monitor.power_samples else 0.0
                print(f"[{now.strftime('%H:%M:%S')}] Tracks: {car_count} ({density}) | VDD_IN: {current_w:.2f}W")

            # ==========================================
            # VISUAL RENDERING BLOCK (Every Frame)
            # ==========================================
            # 1. Draw the transparent green ROI tracking zone boundaries
            cv2.polylines(frame, [ROI_POINTS], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # 2. Draw bounding boxes and text descriptors for verified tracks
            for box, track_id in active_tracks:
                x1, y1, x2, y2 = map(int, box)
                
                # Draw main target box bounding borders
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                
                # Render floating identification flag details
                label = f"Car ID: {track_id}"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

            # 3. Render Dashboard Telemetry Overlays on the frame header corner
            current_w = monitor.power_samples[-1] if monitor.power_samples else 0.0
            cv2.putText(frame, f"System Power: {current_w:.2f} W", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Current Flow: {len(active_tracks)} Cars", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Display the final rendered frame
            cv2.imshow("Car Counting Detection - Edge STM Experiment", frame)
            
            # Use 'q' to break execution or exit cleanly
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except (KeyboardInterrupt, SystemExit):
        print("\nVisual loop closed by user exception.")        
    finally:
        cap.release()
        monitor.stop()
        cv2.destroyAllWindows()
        print("Visual context windows terminated.")                

if __name__ == "__main__":
    main()
