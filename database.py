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
            date      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
    finally:
        conn.close()


def save_detection(filename: str, result: str, confidence: float | None = None):
    """Insert one detection row. Swallows DB errors so a write failure
    never crashes the Flask request."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detections (filename, result, confidence) VALUES (?, ?, ?)",
            (filename, result, confidence),
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"[DB ERROR] Failed to save detection: {e}")
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