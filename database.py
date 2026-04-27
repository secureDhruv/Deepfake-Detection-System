import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections(
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            filename  TEXT,
            result    TEXT,
            confidence REAL,
            date      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details   TEXT
        )
        """)
        
        # Upgrade existing database schema safely.
        cursor.execute("PRAGMA table_info(detections)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'confidence' not in columns:
            cursor.execute("ALTER TABLE detections ADD COLUMN confidence REAL")
        if 'details' not in columns:
            cursor.execute("ALTER TABLE detections ADD COLUMN details TEXT")
            
        conn.commit()
    finally:
        conn.close()


def save_detection(filename: str, result: str, confidence: float | None = None, details: str | None = None):
    """Insert one detection row. Swallows DB errors so a write failure
    never crashes the Flask request."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detections (filename, result, confidence, details) VALUES (?, ?, ?, ?)",
            (filename, result, confidence, details),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[DB ERROR] Failed to save detection: {e}")
        return None
    finally:
        conn.close()


def get_all_detections() -> list:
    """Return all detection rows ordered newest-first."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections ORDER BY id DESC")
        return cursor.fetchall()
    finally:
        conn.close()


def get_detection_by_id(record_id: int):
    """Return a single detection row by its primary key, or None if not found."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections WHERE id = ?", (record_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def delete_detection(record_id: int) -> bool:
    """Delete one detection row and return whether anything was removed."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detections WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def clear_detections() -> None:
    """Delete all detection records."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM detections")
        conn.commit()
    finally:
        conn.close()
