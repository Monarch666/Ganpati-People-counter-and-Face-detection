import numpy as np
import cv2
import uuid
import insightface
import threading
import queue
from db_manager import init_db, load_gallery, enroll_person, log_event


class RecognitionEngine:
    """
    Three-tier person identification engine for high-density crowd counting.
    
    Strategy priority:
      1. ArcFace facial embedding (best accuracy, needs visible face)
      2. Body HSV histogram ReID (fallback when face not visible)
      3. Guaranteed count (never silently drop a person)
    """

    def __init__(self, match_threshold=0.5, body_match_threshold=0.65):
        self.match_threshold = match_threshold
        self.body_match_threshold = body_match_threshold

        # Initialize DB and load gallery into memory
        init_db()
        self.gallery_ids, self.gallery_matrix = load_gallery()
        self._gallery_lock = threading.Lock()

        # Body appearance gallery (session-only, in-memory)
        # Resets on restart because people wear different clothes each day
        self.body_gallery = {}  # {person_id: np.ndarray (HSV histogram)}
        
        # Track unique persons seen in this session
        self.session_unique_persons = set()

        # Initialize InsightFace
        print("Loading InsightFace model...")
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print(f"Engine ready. Loaded {len(self.gallery_ids)} faces into memory.")

        # Initialize Queue and 4 Background Workers for massive throughput (200-300 people)
        self.task_queue = queue.Queue()
        for i in range(4):
            t = threading.Thread(
                target=self._worker, daemon=True, name=f"FaceWorker-{i}"
            )
            t.start()
        print("Background recognition workers started (4 threads).")

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def get_embedding(self, crop):
        """Extracts 512-d ArcFace embedding from a body crop image."""
        if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None
        faces = self.app.get(crop)
        if not faces:
            return None
        return faces[0].embedding

    def get_body_descriptor(self, crop):
        """Extracts a compact HSV color histogram from the upper body.
        
        Uses the upper 60% of the crop (torso) which is more distinctive
        than legs. Returns a 1024-d normalized float32 vector.
        """
        if crop is None or crop.shape[0] < 30 or crop.shape[1] < 20:
            return None
        h = crop.shape[0]
        upper = crop[0 : int(h * 0.6), :]
        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        # 16 hue × 8 sat × 8 val = 1024-d descriptor
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    # =========================================================================
    # Matching
    # =========================================================================

    def match_face(self, new_embedding):
        """Cosine similarity match against the persistent face gallery."""
        with self._gallery_lock:
            if len(self.gallery_ids) == 0:
                return None, 0.0

            norms_gallery = np.linalg.norm(self.gallery_matrix, axis=1)
            norm_new = np.linalg.norm(new_embedding)

            # Guard against zero-norm vectors that would cause div-by-zero
            valid = (norms_gallery > 1e-6) & (norm_new > 1e-6)
            if not np.any(valid):
                return None, 0.0

            sims = np.zeros(len(self.gallery_ids))
            sims[valid] = (self.gallery_matrix[valid] @ new_embedding) / (
                norms_gallery[valid] * norm_new
            )

            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            if best_score >= self.match_threshold:
                return self.gallery_ids[best_idx], best_score
            return None, best_score

    def match_body(self, descriptor):
        """Histogram correlation match against the session body gallery."""
        if not self.body_gallery or descriptor is None:
            return None, 0.0

        best_id = None
        best_score = 0.0
        for person_id, gallery_desc in self.body_gallery.items():
            score = cv2.compareHist(descriptor, gallery_desc, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_id = person_id

        if best_score >= self.body_match_threshold:
            return best_id, float(best_score)
        return None, float(best_score)

    # =========================================================================
    # Queue & Workers
    # =========================================================================

    def _worker(self):
        """Background thread that continuously processes queued crops."""
        while True:
            crops, track_id = self.task_queue.get()
            try:
                res = self.process_entry_crossing(crops, track_id)
                print(f"[FACE_ENGINE] Track {track_id}: {res}")
            except Exception as e:
                print(f"[FACE_ENGINE] Error processing track {track_id}: {e}")
            finally:
                self.task_queue.task_done()

    def enqueue_entry_crossing(self, crops, track_id: int):
        """Non-blocking. Accepts a single crop (np.ndarray) or list of crops."""
        if isinstance(crops, np.ndarray):
            crops = [crops]
        self.task_queue.put((crops, track_id))

    def get_session_unique_count(self) -> int:
        """Returns the true deduplicated count for the current session."""
        return len(self.session_unique_persons)
        
    def reset_session(self):
        """Resets the session state (useful when user resets the counter)."""
        self.body_gallery.clear()
        self.session_unique_persons.clear()

    # =========================================================================
    # Core Processing Pipeline
    # =========================================================================

    def process_entry_crossing(self, crops: list, track_id: int) -> dict:
        """
        Three-tier matching strategy. Guarantees every person who crosses
        the zone boundary is counted in the database, even if their face
        is completely invisible.
        """
        # === STRATEGY 1: Face Recognition (try each crop, take first hit) ===
        for crop in crops:
            embedding = self.get_embedding(crop)
            if embedding is not None:
                matched_id, score = self.match_face(embedding)

                if matched_id:
                    person_id = matched_id
                    status = "duplicate_face"
                else:
                    person_id = enroll_person(embedding)
                    with self._gallery_lock:
                        self.gallery_ids, self.gallery_matrix = load_gallery()
                    status = "new_face"

                # Also store body descriptor for cross-method dedup
                body_desc = self.get_body_descriptor(crop)
                if body_desc is not None:
                    self.body_gallery[person_id] = body_desc

                log_event(person_id=person_id, direction="IN", track_id=track_id)
                self.session_unique_persons.add(person_id)
                return {
                    "status": status,
                    "person_id": person_id,
                    "similarity": score,
                    "method": "face",
                }

        # === STRATEGY 2: Body ReID (face failed on all crops) ===
        for crop in crops:
            body_desc = self.get_body_descriptor(crop)
            if body_desc is not None:
                matched_id, score = self.match_body(body_desc)

                if matched_id:
                    person_id = matched_id
                    status = "duplicate_body"
                else:
                    person_id = str(uuid.uuid4())
                    self.body_gallery[person_id] = body_desc
                    status = "new_body"

                log_event(person_id=person_id, direction="IN", track_id=track_id)
                self.session_unique_persons.add(person_id)
                return {
                    "status": status,
                    "person_id": person_id,
                    "similarity": score,
                    "method": "body",
                }

        # === STRATEGY 3: Guaranteed Count (everything failed) ===
        person_id = str(uuid.uuid4())
        log_event(person_id=person_id, direction="IN", track_id=track_id)
        self.session_unique_persons.add(person_id)
        return {
            "status": "new_unknown",
            "person_id": person_id,
            "similarity": 0.0,
            "method": "none",
        }


# =============================================================================
# Self-Test
# =============================================================================
if __name__ == "__main__":
    import time

    print("Testing Recognition Engine (3-Tier)...")
    engine = RecognitionEngine()

    test_img = cv2.imread("test_face.jpg")
    if test_img is not None:
        print("\n--- Queueing 5 simultaneous crossings (face path) ---")
        start = time.time()
        for i in range(1, 6):
            engine.enqueue_entry_crossing(test_img, track_id=i)
        print(f"YOLO Thread unblocked in {time.time() - start:.4f} seconds!")
        engine.task_queue.join()
        print("Face-path test completed.")
    else:
        print("No test_face.jpg found; skipping face-path test.")

    # Test body-only fallback with a random colored image
    print("\n--- Testing body-only fallback ---")
    dummy = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    engine.enqueue_entry_crossing([dummy], track_id=99)
    engine.task_queue.join()
    print("Body-only fallback test completed.")

    # Test guaranteed count with a tiny unusable crop
    print("\n--- Testing guaranteed count (tiny crop) ---")
    tiny = np.zeros((5, 5, 3), dtype=np.uint8)
    engine.enqueue_entry_crossing([tiny], track_id=100)
    engine.task_queue.join()
    print("Guaranteed count test completed.")
