import sqlite3
import numpy as np
import datetime
import uuid

DB_PATH = "pandal.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL mode for concurrent reads by the dashboard
    conn.execute("PRAGMA journal_mode=WAL;")
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS gallery (
            person_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            first_seen_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            photo_path TEXT
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            direction TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            track_id INTEGER
        )
    ''')
    conn.commit()
    return conn

def load_gallery():
    """Loads all embeddings from the database into a numpy matrix."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT person_id, embedding FROM gallery").fetchall()
    conn.close()
    
    if not rows:
        return [], np.empty((0, 512), dtype=np.float32)
        
    gallery_ids = [r[0] for r in rows]
    # Convert BLOB back to numpy float32 array
    gallery_matrix = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    
    return gallery_ids, gallery_matrix

def enroll_person(embedding, photo_path=None):
    """Adds a new person to the gallery."""
    person_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO gallery (person_id, embedding, photo_path) VALUES (?, ?, ?)",
        (person_id, embedding.tobytes(), photo_path)
    )
    conn.commit()
    conn.close()
    return person_id

def log_event(person_id, direction, track_id):
    """Logs an entry or exit event."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (person_id, direction, track_id) VALUES (?, ?, ?)",
        (person_id, direction, track_id)
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully at", DB_PATH)
