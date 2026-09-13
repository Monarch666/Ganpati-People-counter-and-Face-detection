import argparse
import sys
import time
import json
import os
import threading
import math
import subprocess
from collections import defaultdict

# NOTE: Do NOT set OPENCV_FFMPEG_CAPTURE_OPTIONS env var - it's unreliable on Windows.
# We pass TCP transport directly via VideoCapture() API parameters below.
import cv2
import numpy as np
import insightface
from ultralytics import YOLO
from recognition_engine import RecognitionEngine


class FrameGrabber:
    """
    FFmpeg-piped frame grabber. Uses a ffmpeg subprocess with -rtsp_transport tcp
    to decode HEVC/H.264 RTSP streams without packet loss or reference frame errors.
    Outputs raw BGR24 frames into a numpy buffer at a capped 15 FPS.
    """
    GRAB_WIDTH  = 1280
    GRAB_HEIGHT = 720
    GRAB_FPS    = 15

    def __init__(self, source_str):
        self.source_str = source_str
        self.frame = None
        self.ret   = False
        self.lock  = threading.Lock()
        self.running = True
        self.proc  = None
        self._start()

    def _build_cmd(self):
        is_rtsp = self.source_str.startswith("rtsp://")
        cmd = ["ffmpeg", "-loglevel", "error"]
        if is_rtsp:
            cmd += ["-rtsp_transport", "tcp",
                    "-timeout", "5000000"]  # 5s connect timeout
        cmd += [
            "-i",       self.source_str,
            "-vf",      f"fps={self.GRAB_FPS},scale={self.GRAB_WIDTH}:{self.GRAB_HEIGHT}",
            "-pix_fmt", "bgr24",
            "-vcodec",  "rawvideo",
            "-an",                      # no audio
            "-f",       "rawvideo",
            "pipe:1"
        ]
        return cmd

    def _start(self):
        try:
            self.proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            self.running = True
            t = threading.Thread(target=self._read_loop, daemon=True)
            t.start()
            print(f"[FrameGrabber] ffmpeg started (TCP, {self.GRAB_WIDTH}x{self.GRAB_HEIGHT} @ {self.GRAB_FPS}fps)")
        except Exception as e:
            print(f"[FrameGrabber] Failed to start ffmpeg: {e}")

    def _read_loop(self):
        frame_bytes = self.GRAB_WIDTH * self.GRAB_HEIGHT * 3
        while self.running:
            raw = self.proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                print("[FrameGrabber] Stream ended or corrupted. Restarting in 3s...")
                time.sleep(3)
                self._start()
                return
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.GRAB_HEIGHT, self.GRAB_WIDTH, 3)
            )
            with self.lock:
                self.frame = frame
                self.ret = True

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                pass


class ZoneDrawer:
    """Handles interactive polygon drawing and loading/saving config."""
    def __init__(self, window_name, config_file="zone_config.json"):
        self.window_name = window_name
        self.config_file = config_file
        self.pts = []
        self.done = False

    def load_zone(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    if "polygon" in data and len(data["polygon"]) >= 3:
                        self.pts = [tuple(p) for p in data["polygon"]]
                        return True
            except Exception as e:
                print(f"[Warning] Failed to load zone config: {e}")
        return False

    def save_zone(self):
        if len(self.pts) >= 3:
            with open(self.config_file, "w") as f:
                json.dump({"polygon": self.pts}, f)
            print(f"[Info] Zone saved to {self.config_file}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.done:
                self.pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.pts) >= 3:
                self.done = True

    def draw_interactive(self, frame):
        print("\n--- Zone Drawing Mode ---")
        print("Left-Click : Add point")
        # Ensure we have a named window created
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print("Right-Click / Enter : Finish polygon")
        print("Backspace  : Undo last point")
        print("ESC        : Quit without saving")
        
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while not self.done:
            overlay = frame.copy()
            
            # Draw lines and points
            if len(self.pts) > 0:
                for i in range(len(self.pts) - 1):
                    cv2.line(overlay, self.pts[i], self.pts[i+1], (0, 255, 255), 2)
                for pt in self.pts:
                    cv2.circle(overlay, pt, 4, (0, 0, 255), -1)
                
                # Draw dynamic line to mouse if we wanted to, but simple clicks are fine
                if len(self.pts) >= 3 and self.done:
                    cv2.line(overlay, self.pts[-1], self.pts[0], (0, 255, 255), 2)

            cv2.putText(overlay, "DRAW QUEUE ZONE (Left Click: Add, Right Click: Finish)", 
                        (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(self.window_name, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                sys.exit(0)
            elif key == 8 or key == ord('z'):  # Backspace or Z
                if len(self.pts) > 0:
                    self.pts.pop()
            elif key == 13:  # Enter
                if len(self.pts) >= 3:
                    self.done = True

        cv2.setMouseCallback(self.window_name, lambda *args: None)
        self.save_zone()
        return np.array(self.pts, dtype=np.int32)


class LiveFaceDetector:
    """Lightweight asynchronous Face Detector for on-screen visualization.
    Uses a fast model and skips frames to avoid competing with YOLO for GPU."""
    def __init__(self):
        print("Loading LiveFaceDetector (lightweight scrfd_500m for visualization)...")
        # Use scrfd_500m_bnkps — much lighter than buffalo_l, sufficient for visualization
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_s",   # lightweight: only 4MB vs 300MB for buffalo_l
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection"]  # detection only — skip recognition to save GPU
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))  # smaller det_size = much faster
        self.latest_frame = None
        self.faces = []
        self.lock = threading.Lock()
        self.frame_counter = 0
        self.skip_frames = 4  # Only run detection every 5th frame submitted
        self.thread = threading.Thread(target=self._worker, daemon=True, name="LiveFaceDetector")
        self.thread.start()
        print("LiveFaceDetector ready.")

    def submit_frame(self, frame):
        """Non-blocking frame submission. Skips frames to cap GPU usage."""
        self.frame_counter += 1
        if self.frame_counter % (self.skip_frames + 1) != 0:
            return  # skip this frame
        with self.lock:
            if self.latest_frame is None:  # only update if worker has finished previous
                self.latest_frame = frame.copy()

    def get_faces(self):
        """Returns the latest cached face bounding boxes (thread-safe)."""
        with self.lock:
            return list(self.faces)

    def _worker(self):
        while True:
            with self.lock:
                frame = self.latest_frame
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                faces = self.app.get(frame)
                bboxes = [f.bbox for f in faces]
            except Exception:
                bboxes = []
            with self.lock:
                self.faces = bboxes
                self.latest_frame = None  # signal worker is idle


def parse_args():
    parser = argparse.ArgumentParser(description="Dense Crowd Zone-Based People Counter")
    parser.add_argument("--source", type=str, default="0", help="Video source (webcam index, RTSP url, video file)")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLO model (default: yolov8m.pt for dense crowds)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution (default: 1280)")
    parser.add_argument("--max-det", type=int, default=1000, help="Max detections per frame (default: 1000)")
    parser.add_argument("--tracker", type=str, default="crowd_bytetrack.yaml", help="Custom tracker config")
    parser.add_argument("--zone-file", type=str, default="zone_config.json", help="Path to zone config JSON")
    parser.add_argument("--redraw-zone", action="store_true", help="Force redrawing the zone on startup")
    parser.add_argument("--flip", action="store_true", help="Flip the feed horizontally")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame to save CPU (default: 1)")
    return parser.parse_args()


def draw_hud(frame, total_entered, total_exited, live_occupancy, fps, active_tracks, zone_poly):
    """Draw the polygon zone and the heads-up display overlay."""
    h, w = frame.shape[:2]
    
    # Draw Polygon Zone with translucent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_poly], (255, 100, 100))
    cv2.polylines(overlay, [zone_poly], isClosed=True, color=(0, 255, 255), thickness=3)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # Top banner background
    banner_height = 65
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 24), -1)

    # Bottom banner background
    helper_height = 30
    cv2.rectangle(overlay, (0, h - helper_height), (w, h), (20, 20, 24), -1)

    # Blend UI banners
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Enter / Exit / Inside
    cv2.putText(frame, f"TOTAL COUNTED: {total_entered}", (20, 42), font, 0.75, (50, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, f"PROCESSING ZONE: {live_occupancy}", (430, 42), font, 0.75, (240, 220, 60), 2, cv2.LINE_AA)

    # Stats
    stats_text = f"FPS: {fps:.1f} | Active: {active_tracks}"
    (tw, _), _ = cv2.getTextSize(stats_text, font, 0.55, 1)
    cv2.putText(frame, stats_text, (w - tw - 20, 40), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Controls Helper
    cv2.putText(frame, "Hotkeys: [Q] Quit  |  [R] Reset  |  [D] Redraw Zone  |  [T] Toggle Trails", 
                (15, h - 9), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    print("=" * 60)
    print("   Zone-Based Dense Queue Counting (YOLOv8 + ByteTrack)")
    print("=" * 60)
    
    print("Initializing Recognition Engine...")
    engine = RecognitionEngine()
    live_detector = LiveFaceDetector()
    
    print(f"Loading model: {args.model} (imgsz={args.imgsz}, max_det={args.max_det})")
    
    model = YOLO(args.model)

    print(f"Opening video source: {args.source}...")
    grabber = FrameGrabber(args.source)
    if not grabber.running:
        print("[ERROR] Failed to open video source.")
        sys.exit(1)

    window_name = "People Counter - Zone Based"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Wait for the first frame
    ret, frame = grabber.read()
    while not ret or frame is None:
        time.sleep(0.1)
        ret, frame = grabber.read()

    if args.flip:
        frame = cv2.flip(frame, 1)
        
    cv2.resizeWindow(window_name, min(1280, frame.shape[1]), min(720, frame.shape[0]))

    # Setup Polygon Zone
    zone_drawer = ZoneDrawer(window_name, args.zone_file)
    if args.redraw_zone or not zone_drawer.load_zone():
        zone_poly = zone_drawer.draw_interactive(frame)
    else:
        zone_poly = np.array(zone_drawer.pts, dtype=np.int32)
        print(f"[Info] Loaded existing zone from {args.zone_file}")

    # State variables
    tracks_inside = set()       # IDs currently inside the polygon
    track_last_seen = {}        # ID -> frame_index (for stale cleanup)
    track_history = defaultdict(list) # ID -> list of (cx, cy) points for drawing trails
    
    total_entered = 0  # Not strictly used on HUD anymore, but kept for state
    total_exited = 0   # Not strictly used on HUD anymore, but kept for state
    draw_trails = True
    counted_tracks = set()      # IDs successfully processed/counted (prevents double processing)
    track_crops = {}            # ID -> list of body crops (multi-frame evidence)

    fps = 0.0
    frame_idx = 0
    start_time = time.time()
    frames_processed = 0

    while True:
        ret, frame = grabber.read()
        if not ret or frame is None:
            # Don't crash - just skip this frame and wait for reconnect
            time.sleep(0.1)
            continue
            
        frame_idx += 1
        if frame_idx % args.skip_frames != 0:
            continue
            
        if args.flip:
            frame = cv2.flip(frame, 1)

        # Track objects
        results = model.track(
            frame,
            persist=True,
            classes=[0],           # only persons
            conf=args.conf,
            iou=0.60,              # Lower IoU allows detecting highly overlapping people in dense queues
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=False,
            tracker=args.tracker
        )

        active_tracks = 0
        current_frame_ids = set()

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            active_tracks = len(track_ids)
            
            # Optimization: turn off trails if crowd is massive to save drawing CPU
            is_dense = active_tracks > 100
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                current_frame_ids.add(track_id)
                track_last_seen[track_id] = frame_idx
                
                # Bottom-center feet anchor
                feet_x = float((x1 + x2) / 2.0)
                feet_y = float(y2)
                
                # --- Multi-frame crop accumulation ---
                # We want to collect the best, largest crops of the person.
                # If they are not yet counted, we keep upgrading the best crops.
                if track_id not in counted_tracks:
                    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)].copy()
                    crop_area = crop.shape[0] * crop.shape[1]
                    if crop_area > 500:  # Minimum viable crop size
                        if track_id not in track_crops:
                            track_crops[track_id] = []
                        crops_list = track_crops[track_id]
                        if len(crops_list) < 5:
                            crops_list.append(crop)
                        else:
                            # Replace the smallest crop with this one if larger
                            areas = [c.shape[0] * c.shape[1] for c in crops_list]
                            min_idx = int(np.argmin(areas))
                            if crop_area > areas[min_idx]:
                                crops_list[min_idx] = crop

                # Check polygon inclusion (>= 0 means inside)
                is_inside = cv2.pointPolygonTest(zone_poly, (feet_x, feet_y), False) >= 0
                
                # State transitions
                if is_inside:
                    if track_id not in tracks_inside:
                        tracks_inside.add(track_id)
                        print(f"[TRACKING] ID {track_id} entered zone (collecting evidence...)")
                else:
                    if track_id in tracks_inside:
                        tracks_inside.remove(track_id)
                        
                        # Process upon EXIT instead of ENTRY!
                        if track_id not in counted_tracks:
                            counted_tracks.add(track_id)
                            print(f"[EVENT] ID {track_id} EXITED zone. Submitting to Recognition Engine.")
                            
                            crops_to_send = track_crops.pop(track_id, None)
                            if not crops_to_send:
                                crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)].copy()
                                crops_to_send = [crop] if crop.shape[0] > 0 and crop.shape[1] > 0 else []
                            if crops_to_send:
                                engine.enqueue_entry_crossing(crops_to_send, track_id)

                # Drawing boxes and feet anchors
                box_color = (0, 255, 128) if is_inside else (255, 180, 0)
                box_thickness = 1 if is_dense else 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, box_thickness)
                
                # Only draw feet anchors and labels if not a massive crowd (to prevent visual clutter)
                if not is_dense:
                    cv2.circle(frame, (int(feet_x), int(feet_y)), 5, (0, 0, 255), -1)  # Feet anchor
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), max(int(y1) - 8, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

                # Trail logic
                if draw_trails and not is_dense:
                    cx = int((x1 + x2) / 2.0)
                    cy = int((y1 + y2) / 2.0)
                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > 15:
                        track_history[track_id].pop(0)
                        
                    points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 200, 255), thickness=2)

        # Cleanup stale tracks (ID exits camera completely while inside zone)
        # Prevents ghost occupancies if someone disappears behind an occlusion forever
        if frame_idx % 30 == 0:
            stale_threshold = 90  # frames
            stale_ids = [tid for tid, last_frame in track_last_seen.items() if (frame_idx - last_frame) > stale_threshold]
            for tid in stale_ids:
                if tid in tracks_inside:
                    tracks_inside.remove(tid)
                    
                    # They disappeared while inside. We treat this as an EXIT and process them!
                    if tid not in counted_tracks:
                        counted_tracks.add(tid)
                        print(f"[CLEANUP] ID {tid} dropped inside zone. Auto-submitting to engine.")
                        crops_to_send = track_crops.pop(tid, None)
                        if crops_to_send:
                            engine.enqueue_entry_crossing(crops_to_send, tid)
                del track_last_seen[tid]
                if tid in track_history:
                    del track_history[tid]
                track_crops.pop(tid, None)

        live_occupancy = len(tracks_inside)
        
        # Calculate FPS periodically
        frames_processed += 1
        now = time.time()
        elapsed = now - start_time
        if elapsed >= 0.5:
            fps = frames_processed / elapsed
            frames_processed = 0
            start_time = now

        # --- Draw Live Face Bounding Boxes ---
        live_detector.submit_frame(frame)  # Non-blocking, skips frames automatically
        current_faces = live_detector.get_faces()
        for bbox in current_faces:
            x1_f, y1_f, x2_f, y2_f = map(int, bbox)
            cv2.rectangle(frame, (x1_f, y1_f), (x2_f, y2_f), (255, 0, 255), 2)
            cv2.putText(frame, "FACE", (x1_f, y1_f - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

        # Get true deduplicated count from the recognition engine
        true_processed_count = engine.get_session_unique_count()

        draw_hud(frame, true_processed_count, 0, live_occupancy, fps, active_tracks, zone_poly)

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            total_exited = 0
            tracks_inside.clear()
            track_history.clear()
            counted_tracks.clear()
            track_crops.clear()
            engine.reset_session()
            print("[RESET] Counts and states reset.")
        elif key in (ord('t'), ord('T')):
            draw_trails = not draw_trails
            print(f"[CONFIG] Trails drawing: {draw_trails}")
        elif key in (ord('d'), ord('D')):
            print("[CONFIG] Redrawing zone...")
            zone_drawer.pts = []
            zone_drawer.done = False
            cv2.destroyWindow(window_name)
            zone_poly = zone_drawer.draw_interactive(grabber.read()[1])
            tracks_inside.clear() # Reset state since zone changed
            total_exited = 0
            counted_tracks.clear()
            track_crops.clear()
            engine.reset_session()
            print("[INFO] Zone updated, counts reset.")

    grabber.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()
