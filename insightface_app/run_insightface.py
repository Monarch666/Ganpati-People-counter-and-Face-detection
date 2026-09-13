import cv2
import time
import onnxruntime
from insightface.app import FaceAnalysis

def main():
    print("Initializing InsightFace (this will download model weights on the first run)...")
    # Initialize the FaceAnalysis app. 
    # 'buffalo_l' is a commonly used comprehensive model pack for face detection, landmark, and recognition.
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    # Changed to CUDAExecutionProvider to run on GPU and relieve CPU load.
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # --- GPU Diagnostics ---
    print("=" * 60)
    print("           InsightFace GPU Webcam Test")
    print("=" * 60)
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    print(f"Available providers:  {onnxruntime.get_available_providers()}")
    
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        print("[OK] CUDAExecutionProvider is available!")
    else:
        print("[WARNING] CUDAExecutionProvider NOT available. Will fall back to CPU.")
    print()

    print("Initializing InsightFace on GPU...")
    
    # Force CUDA with explicit device configuration
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit (GTX 1650 has 4GB)
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
        }),
        'CPUExecutionProvider',  # Fallback
    ]
    
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app = FaceAnalysis(name='buffalo_s', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Verify which provider each model is actually using
    print("\nModel session providers:")
    for model in app.models.values():
        if hasattr(model, 'session'):
            actual_providers = model.session.get_providers()
            print(f"  {model.__class__.__name__}: {actual_providers}")
    print()

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")
    print("=" * 60)
    
    # FPS tracking
    fps = 0.0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # InsightFace expects BGR images (which OpenCV provides by default)
        faces = app.get(frame)
        
        # Draw bounding boxes and landmarks
        for face in faces:
            box = face.bbox.astype(int)
            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw landmarks (5 facial points)
            if face.kps is not None:
                for kp in face.kps:
                    kp = kp.astype(int)
                    cv2.circle(frame, tuple(kp), 2, (0, 0, 255), -1)
                    
            # Draw age/gender if available
            text = f"{'M' if face.gender==1 else 'F'} {face.age}"
            cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        
        # FPS calculation
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        
        # Draw FPS and GPU indicator
        cv2.putText(frame, f"FPS: {fps:.1f} | GPU: CUDA (GTX 1650)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("InsightFace - Webcam Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Closed InsightFace application.")

if __name__ == '__main__':
    main()

