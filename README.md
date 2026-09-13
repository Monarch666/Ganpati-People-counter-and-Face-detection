# Ganpati People Counter & Face Detection

This repository contains a dual-purpose AI computer vision system designed for high-density crowd tracking and real-time facial analysis. It is optimized to run on local NVIDIA GPUs using ONNX Runtime and PyTorch.

## 🚀 Features

### 1. Web Dashboard for High-Density Queue Counting
A FastAPI-based local web dashboard that streams live video from IP cameras (RTSP) and tracks people entering or exiting a custom-defined polygon zone. 
- **Core Stack:** YOLOv8n + ByteTrack + OpenCV + FastAPI
- **Optimizations:** Configured for high FPS on consumer GPUs (e.g., GTX 1650). It uses `imgsz=640`, processes every alternate frame, and defaults to camera sub-streams to maintain a smooth 15-25+ FPS.
- **Zone Based:** Uses "feet-anchored" counting to accurately measure entries and exits without perspective distortion.

### 2. InsightFace Facial Detection & Demographics
A lightweight script powered by InsightFace's `buffalo_s` model to detect faces, 5-point landmarks, age, and gender via webcam.
- **Hardware Acceleration:** Fully configured to use `CUDAExecutionProvider` via `onnxruntime-gpu`.

---

## 📋 Requirements

- Python 3.10 or higher
- NVIDIA GPU with CUDA 12.x support (e.g., GTX 1650, RTX 2000 Ada)
- **Required Libraries:**
  ```bash
  pip install ultralytics opencv-python fastapi uvicorn insightface onnxruntime-gpu==1.28.0
  ```

---

## 🛠️ How to Run

### Running the People Counter Dashboard
1. Open the main folder.
2. Double-click the **`run_dashboard.bat`** file.
3. Once the terminal says the server is running, open your web browser and go to:
   👉 **http://localhost:5002**
4. *To change the camera IP or credentials, right-click `run_dashboard.bat` and edit the `--ip`, `--user`, and `--password` flags.*

### Running the Face Detection (Webcam)
1. Open the `insightface_app` folder.
2. Double-click the **`run_insightface.bat`** file.
3. A terminal will open (and download the lightweight ONNX models if it's your first run). 
4. A video window will pop up showing real-time face detection, age, and gender from your webcam. Press `q` to quit.

---

## ⚙️ Configuration

- **`crowd_bytetrack.yaml`**: Contains the custom tracker thresholds. The track buffer is artificially inflated to handle heavy occlusion in dense crowds.
- **`web_dashboard.py`**: The main FastAPI application. You can adjust the `imgsz`, `--skip-frames`, and `--model` arguments here if you upgrade your GPU (e.g., moving to an RTX 2000 Ada allows switching back to `yolov8s` at `1280p`).
- **`zone_config.json`**: Automatically generated file that saves your custom polygon boundaries.

## 👤 Author
Developed by **[Monarch666](https://github.com/Monarch666)**
