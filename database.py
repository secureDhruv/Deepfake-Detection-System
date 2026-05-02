import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "history.db")


def _search_filter(search: str | None) -> tuple[str, tuple[str, ...]]:
    if not search:
        return "", ()

    query = f"%{search.lower()}%"
    return "WHERE lower(filename) LIKE ?", (query,)


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


def get_all_detections(limit: int | None = None, offset: int = 0, search: str | None = None) -> list:
    """Return all detection rows ordered newest-first."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        where_clause, params = _search_filter(search)
        if limit is None:
            cursor.execute(f"SELECT * FROM detections {where_clause} ORDER BY id DESC", params)
        else:
            cursor.execute(
                f"SELECT * FROM detections {where_clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            )
        return cursor.fetchall()
    finally:
        conn.close()


def get_detection_count(search: str | None = None) -> int:
    """Return the total number of stored detection rows."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        where_clause, params = _search_filter(search)
        cursor.execute(f"SELECT COUNT(*) FROM detections {where_clause}", params)
        return int(cursor.fetchone()[0] or 0)
    finally:
        conn.close()


def get_detection_stats(search: str | None = None) -> dict:
    """Return aggregate detection statistics without loading every row."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        where_clause, params = _search_filter(search)
        cursor.execute(
            f"""
            SELECT
                COUNT(*),
                SUM(CASE WHEN lower(result) LIKE '%fake%' THEN 1 ELSE 0 END),
                AVG(confidence),
                SUM(CASE WHEN confidence > 75 THEN 1 ELSE 0 END),
                SUM(CASE WHEN confidence IS NOT NULL AND confidence <= 25 THEN 1 ELSE 0 END),
                SUM(CASE WHEN confidence > 25 AND confidence <= 50 THEN 1 ELSE 0 END),
                SUM(CASE WHEN confidence > 50 AND confidence <= 75 THEN 1 ELSE 0 END),
                SUM(CASE WHEN confidence > 75 THEN 1 ELSE 0 END)
            FROM detections
            {where_clause}
            """,
            params,
        )
        (
            total,
            fake_count,
            avg_conf,
            high_conf_count,
            dist_0_25,
            dist_25_50,
            dist_50_75,
            dist_75_100,
        ) = cursor.fetchone()
    finally:
        conn.close()

    total = int(total or 0)
    fake_count = int(fake_count or 0)
    real_count = total - fake_count
    high_conf_count = int(high_conf_count or 0)

    fake_pct = (fake_count / total * 100) if total else 0
    real_pct = (real_count / total * 100) if total else 0
    high_conf_pct = (high_conf_count / total * 100) if total else 0

    return {
        "total": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "avg_conf": round(avg_conf or 0, 1),
        "fake_pct": fake_pct,
        "real_pct": real_pct,
        "dist": {
            "0-25": int(dist_0_25 or 0),
            "25-50": int(dist_25_50 or 0),
            "50-75": int(dist_50_75 or 0),
            "75-100": int(dist_75_100 or 0),
        },
        "high_conf_pct": high_conf_pct,
    }


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
