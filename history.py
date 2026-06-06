import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.getenv("PIXELTRUTH_DB_PATH", "pixeltruth_history.db")

def init_db(db_path=DB_PATH):
    """
    Initializes the SQLite database and creates the predictions table if it doesn't exist.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence_pct REAL NOT NULL,
                face_detected INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()

def save_prediction(filename, verdict, confidence_pct, face_detected, db_path=DB_PATH):
    """
    Saves a single prediction to the database.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # UTC ISO-8601 (CWE-840 fix)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (timestamp, filename, verdict, confidence_pct, face_detected)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, filename, verdict, confidence_pct, face_detected))
        conn.commit()

def load_history(limit=200, db_path=DB_PATH):
    """
    Loads the prediction history from the database, newest first.
    Returns rows as dicts matching the existing session_state dict structure.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT filename, verdict, confidence_pct, timestamp, face_detected
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()

    history = []
    for row in rows:
        history.append({
            "Filename": row[0],
            "Result": row[1],
            "Confidence (%)": f"{row[2]:.1f}",
            "Timestamp": row[3],
            "Face Detected": bool(row[4])
        })
    return history

def clear_history(db_path=DB_PATH):
    """
    Clears all prediction history from the database.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
