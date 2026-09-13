import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO


# ──────────────────────────────────────────────
#  Global shared state (updated by tracker thread)
# ──────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None               # Latest annotated frame (BGR)
        self.total_entered = 0
        self.total_exited = 0
        self.live_occupancy = 0
        self.active_tracks = 0
        self.fps = 0.0
        self.running = True
        self.zone_poly = None            # np.array of polygon points
        self.source_url = ""
        self.stream_connected = False

state = AppState()


# ──────────────────────────────────────────────
#  Zone config helpers
# ──────────────────────────────────────────────
def load_zone(config_file="zone_config.json"):
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
                if "polygon" in data and len(data["polygon"]) >= 3:
                    return np.array(data["polygon"], dtype=np.int32)
        except Exception:
            pass
    return None


def save_zone(polygon, config_file="zone_config.json"):
    pts = polygon.tolist() if isinstance(polygon, np.ndarray) else polygon
    with open(config_file, "w") as f:
        json.dump({"polygon": pts}, f)


# ──────────────────────────────────────────────
#  Tracking thread — runs the YOLOv8 + ByteTrack
#  pipeline continuously and pushes annotated
#  frames + stats into the shared AppState.
# ──────────────────────────────────────────────
def tracker_thread(args):
    global state

    print(f"[Tracker] Loading model: {args.model}")
    model = YOLO(args.model)

    rtsp_urls = [
        # Sub-streams first (lower resolution = much faster processing)
        f"rtsp://{args.user}:{args.password}@{args.ip}:554/Streaming/Channels/102",
        f"rtsp://{args.user}:{args.password}@{args.ip}:554/cam/realmonitor?channel=1&subtype=1",
        # Main streams as fallback
        f"rtsp://{args.user}:{args.password}@{args.ip}:554/Streaming/Channels/101",
        f"rtsp://{args.user}:{args.password}@{args.ip}:554/cam/realmonitor?channel=1&subtype=0",
    ]

    if args.source:
        rtsp_urls.insert(0, args.source)

    cap = None
    for url in rtsp_urls:
        print(f"[Tracker] Trying RTSP: {url}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"[Tracker] Connected successfully to: {url}")
            with state.lock:
                state.source_url = url
                state.stream_connected = True
            break
        else:
            cap.release()
            cap = None
            print(f"[Tracker] Failed: {url}")

    if cap is None:
        print("[Tracker] ERROR: Could not connect to any RTSP URL.")
        print("[Tracker] Please verify camera IP, credentials, and network.")
        with state.lock:
            state.stream_connected = False
        return

    # Load or create a default zone
    zone_poly = load_zone(args.zone_file)
    if zone_poly is None:
        # Create a default rectangular zone covering center 60% of frame
        ret, first_frame = cap.read()
        if ret and first_frame is not None:
            h, w = first_frame.shape[:2]
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            zone_poly = np.array([
                [margin_x, margin_y],
                [w - margin_x, margin_y],
                [w - margin_x, h - margin_y],
                [margin_x, h - margin_y]
            ], dtype=np.int32)
            save_zone(zone_poly, args.zone_file)
            print(f"[Tracker] Created default zone (center 60%). Edit via dashboard or {args.zone_file}")
        else:
            zone_poly = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.int32)

    with state.lock:
        state.zone_poly = zone_poly

    # Tracking state
    tracks_inside = set()
    track_last_seen = {}
    track_history = defaultdict(list)
    total_entered = 0
    total_exited = 0

    frame_idx = 0
    fps_count = 0
    fps_start = time.time()
    fps = 0.0

    print("[Tracker] Starting tracking loop...")

    while state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            # Reconnect
            print("[Tracker] Stream dropped, reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(state.source_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame_idx += 1
        if frame_idx % args.skip_frames != 0:
            continue

        # Run YOLO tracking
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=args.conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=False,
            tracker=args.tracker,
        )

        active_tracks = 0
        current_frame_ids = set()

        with state.lock:
            zone_poly = state.zone_poly

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            active_tracks = len(track_ids)
            is_dense = active_tracks > 100

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                current_frame_ids.add(track_id)
                track_last_seen[track_id] = frame_idx

                feet_x = float((x1 + x2) / 2.0)
                feet_y = float(y2)

                is_inside = cv2.pointPolygonTest(zone_poly, (feet_x, feet_y), False) >= 0

                if is_inside:
                    if track_id not in tracks_inside:
                        total_entered += 1
                        tracks_inside.add(track_id)
                else:
                    if track_id in tracks_inside:
                        total_exited += 1
                        tracks_inside.remove(track_id)

                # Draw on frame
                box_color = (0, 255, 128) if is_inside else (255, 180, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.circle(frame, (int(feet_x), int(feet_y)), 4, (0, 0, 255), -1)

                if not is_dense:
                    cv2.putText(frame, f"ID:{track_id}", (int(x1), max(int(y1) - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

        # Stale track cleanup
        if frame_idx % 30 == 0:
            stale_ids = [tid for tid, lf in track_last_seen.items() if (frame_idx - lf) > 90]
            for tid in stale_ids:
                if tid in tracks_inside:
                    total_exited += 1
                    tracks_inside.remove(tid)
                del track_last_seen[tid]
                track_history.pop(tid, None)

        live_occupancy = len(tracks_inside)

        # Draw zone overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_poly], (255, 100, 100))
        cv2.polylines(overlay, [zone_poly], True, (0, 255, 255), 3)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 0.5:
            fps = fps_count / elapsed
            fps_count = 0
            fps_start = time.time()

        # HUD on frame
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 24), -1)
        cv2.putText(frame, f"ENTERED: {total_entered}", (15, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 220, 80), 2, cv2.LINE_AA)
        cv2.putText(frame, f"EXITED: {total_exited}", (220, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (60, 80, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f"INSIDE: {live_occupancy}", (410, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 220, 60), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f} | Tracks: {active_tracks}", (w - 280, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Push to shared state
        with state.lock:
            state.frame = frame.copy()
            state.total_entered = total_entered
            state.total_exited = total_exited
            state.live_occupancy = live_occupancy
            state.active_tracks = active_tracks
            state.fps = fps

    cap.release()
    print("[Tracker] Stopped.")


# ──────────────────────────────────────────────
#  FastAPI Web App
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="People Counter Web Dashboard")
    parser.add_argument("--ip", type=str, default="192.168.1.35", help="Camera IP address")
    parser.add_argument("--user", type=str, default="admin", help="Camera username")
    parser.add_argument("--password", type=str, default="123456", help="Camera password")
    parser.add_argument("--source", type=str, default="", help="Override full RTSP URL (optional)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--max-det", type=int, default=1000, help="Max detections per frame")
    parser.add_argument("--tracker", type=str, default="crowd_bytetrack.yaml", help="Tracker config")
    parser.add_argument("--zone-file", type=str, default="zone_config.json", help="Zone config file")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--port", type=int, default=5002, help="Web dashboard port")
    return parser.parse_args()


args = parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start tracker in background thread
    t = threading.Thread(target=tracker_thread, args=(args,), daemon=True)
    t.start()
    yield
    state.running = False


app = FastAPI(title="People Counter Dashboard", lifespan=lifespan)


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>People Counter — Live Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f0f14;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 18px 30px;
            border-bottom: 2px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 22px;
            color: #e0e0e0;
            font-weight: 600;
        }
        .header h1 span { color: #00d4ff; }
        .header .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        .status-dot {
            width: 10px; height: 10px; border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.live { background: #00ff88; }
        .status-dot.offline { background: #ff4444; animation: none; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 20px;
            padding: 20px 30px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .video-panel {
            background: #1a1a2e;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #2a2a4a;
        }
        .video-panel img {
            width: 100%;
            display: block;
            border-radius: 12px;
        }
        .sidebar { display: flex; flex-direction: column; gap: 16px; }
        .card {
            background: linear-gradient(145deg, #1a1a2e, #16213e);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2a2a4a;
        }
        .card h3 {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #888;
            margin-bottom: 10px;
        }
        .stat-value {
            font-size: 48px;
            font-weight: 700;
            line-height: 1.1;
        }
        .stat-entered .stat-value { color: #00ff88; }
        .stat-exited .stat-value { color: #ff6b6b; }
        .stat-inside .stat-value { color: #00d4ff; }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #2a2a4a;
            font-size: 14px;
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-row .label { color: #888; }
        .stat-row .value { color: #e0e0e0; font-weight: 600; }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
            width: 100%;
        }
        .btn-reset {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }
        .btn-reset:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(231,76,60,0.3); }

        @media (max-width: 900px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 <span>People Counter</span> — Live Dashboard</h1>
        <div class="status">
            <div class="status-dot live" id="statusDot"></div>
            <span id="statusText">Connecting...</span>
        </div>
    </div>

    <div class="container">
        <div class="video-panel">
            <img id="videoFeed" src="/video_feed" alt="Live Video Feed"
                 onerror="this.style.opacity=0.3; document.getElementById('statusDot').className='status-dot offline'; document.getElementById('statusText').textContent='Stream Offline';"
                 onload="document.getElementById('statusDot').className='status-dot live'; document.getElementById('statusText').textContent='LIVE';">
        </div>

        <div class="sidebar">
            <div class="card stat-entered">
                <h3>👥 Total Entered</h3>
                <div class="stat-value" id="entered">0</div>
            </div>
            <div class="card stat-exited">
                <h3>🚶 Total Exited</h3>
                <div class="stat-value" id="exited">0</div>
            </div>
            <div class="card stat-inside">
                <h3>📍 Inside Now</h3>
                <div class="stat-value" id="inside">0</div>
            </div>
            <div class="card">
                <h3>📊 System Stats</h3>
                <div class="stat-row">
                    <span class="label">FPS</span>
                    <span class="value" id="fps">0</span>
                </div>
                <div class="stat-row">
                    <span class="label">Active Tracks</span>
                    <span class="value" id="tracks">0</span>
                </div>
                <div class="stat-row">
                    <span class="label">Camera</span>
                    <span class="value" id="camera">Connecting...</span>
                </div>
            </div>
            <button class="btn btn-reset" onclick="resetCounters()">🔄 Reset All Counts</button>
        </div>
    </div>

    <script>
        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('entered').textContent = data.total_entered;
                document.getElementById('exited').textContent = data.total_exited;
                document.getElementById('inside').textContent = data.live_occupancy;
                document.getElementById('fps').textContent = data.fps.toFixed(1);
                document.getElementById('tracks').textContent = data.active_tracks;
                document.getElementById('camera').textContent = data.connected ? 'Connected' : 'Offline';
            } catch(e) {}
        }

        async function resetCounters() {
            if (confirm('Reset all counters to zero?')) {
                await fetch('/api/reset', { method: 'POST' });
            }
        }

        setInterval(updateStats, 500);
        updateStats();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


def generate_mjpeg():
    """Generator that yields MJPEG frames from the shared state."""
    while state.running:
        with state.lock:
            frame = state.frame
        if frame is not None:
            # Resize for streaming if frame is large (saves encoding time)
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960 / w
                frame = cv2.resize(frame, (960, int(h * scale)))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stats")
async def get_stats():
    with state.lock:
        return JSONResponse({
            "total_entered": state.total_entered,
            "total_exited": state.total_exited,
            "live_occupancy": state.live_occupancy,
            "active_tracks": state.active_tracks,
            "fps": round(state.fps, 1),
            "connected": state.stream_connected,
        })


@app.post("/api/reset")
async def reset_counts():
    with state.lock:
        state.total_entered = 0
        state.total_exited = 0
        state.live_occupancy = 0
    print("[Dashboard] Counters reset via web UI.")
    return JSONResponse({"status": "ok"})


@app.post("/api/zone")
async def update_zone(request: Request):
    data = await request.json()
    if "polygon" in data and len(data["polygon"]) >= 3:
        poly = np.array(data["polygon"], dtype=np.int32)
        with state.lock:
            state.zone_poly = poly
        save_zone(poly, args.zone_file)
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "error", "message": "Need at least 3 points"}, status_code=400)


if __name__ == "__main__":
    print("=" * 60)
    print("   People Counter — Web Dashboard")
    print("=" * 60)
    print(f"   Camera:    {args.ip} ({args.user})")
    print(f"   Model:     {args.model} (imgsz={args.imgsz})")
    print(f"   Dashboard: http://localhost:{args.port}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=args.port)

