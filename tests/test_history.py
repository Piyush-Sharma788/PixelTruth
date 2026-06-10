import gc
import os
import sqlite3
import tempfile
import time
import pytest
from pathlib import Path

from history import init_db, save_prediction, load_history, clear_history

@pytest.fixture
def temp_db():
    """Provides a temporary database path for tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    # Initialize the table for tests
    init_db(db_path=path)
    
    yield path
    
    # Clean up after tests — retry to handle Windows SQLite file lock
    gc.collect()
    for _ in range(5):
        try:
            os.unlink(path)
            break
        except PermissionError:
            time.sleep(0.1)

def test_init_db_creates_table(temp_db):
    # Verify that the 'predictions' table exists
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        table = cursor.fetchone()
        assert table is not None
        assert table[0] == 'predictions'

def test_save_and_load_roundtrip(temp_db):
    save_prediction(
        filename="test_image.jpg",
        verdict="Real",
        confidence_pct=95.5,
        face_detected=1,
        db_path=temp_db
    )
    
    history = load_history(db_path=temp_db)
    assert len(history) == 1
    assert history[0]["Filename"] == "test_image.jpg"
    assert history[0]["Result"] == "Real"
    assert history[0]["Confidence (%)"] == "95.5"
    assert history[0]["Face Detected"] is True
    assert "Timestamp" in history[0]

def test_load_returns_newest_first(temp_db):
    save_prediction("img1.jpg", "Real", 90.0, 1, db_path=temp_db)
    save_prediction("img2.jpg", "Fake", 95.0, 1, db_path=temp_db)
    save_prediction("img3.jpg", "Real", 99.9, 0, db_path=temp_db)
    
    history = load_history(db_path=temp_db)
    assert len(history) == 3
    assert history[0]["Filename"] == "img3.jpg"
    assert history[1]["Filename"] == "img2.jpg"
    assert history[2]["Filename"] == "img1.jpg"

def test_clear_empties_table(temp_db):
    save_prediction("img1.jpg", "Real", 90.0, 1, db_path=temp_db)
    assert len(load_history(db_path=temp_db)) == 1
    
    clear_history(db_path=temp_db)
    assert len(load_history(db_path=temp_db)) == 0

def test_load_limit_is_respected(temp_db):
    for i in range(10):
        save_prediction(f"img{i}.jpg", "Real", 90.0, 1, db_path=temp_db)
        
    history = load_history(limit=5, db_path=temp_db)
    assert len(history) == 5
    # The newest 5 would be img9 down to img5
    assert history[0]["Filename"] == "img9.jpg"
    assert history[-1]["Filename"] == "img5.jpg"

def test_save_and_load_with_hash(temp_db):
    save_prediction(
        filename="test_hash.jpg",
        verdict="Fake",
        confidence_pct=88.2,
        face_detected=0,
        image_hash="abc123xyz",
        db_path=temp_db
    )
    history = load_history(db_path=temp_db)
    assert len(history) == 1
    assert history[0]["Filename"] == "test_hash.jpg"
    assert history[0]["Result"] == "Fake"
    assert history[0]["_hash"] == "abc123xyz"

def test_db_migration_backward_compatibility():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        # Create table with old schema manually
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence_pct REAL NOT NULL,
                    face_detected INTEGER NOT NULL DEFAULT 0
                )
            """)
            cursor.execute("""
                INSERT INTO predictions (timestamp, filename, verdict, confidence_pct, face_detected)
                VALUES ('2026-06-08T12:00:00Z', 'old_img.jpg', 'Real', 95.0, 1)
            """)
            conn.commit()
            
        # Test loading from old schema without running init_db (migration)
        # It should handle OperationalError when selecting image_hash and fallback
        history = load_history(db_path=path)
        assert len(history) == 1
        assert history[0]["Filename"] == "old_img.jpg"
        assert "_hash" not in history[0]
        
        # Now run init_db, which should run the migration to add the column
        init_db(db_path=path)
        
        # Verify that we can save a prediction with a hash now
        save_prediction(
            filename="new_img.jpg",
            verdict="Fake",
            confidence_pct=80.0,
            face_detected=0,
            image_hash="new_hash_123",
            db_path=path
        )
        
        history_after = load_history(db_path=path)
        assert len(history_after) == 2
        
        # Newest first
        assert history_after[0]["Filename"] == "new_img.jpg"
        assert history_after[0]["_hash"] == "new_hash_123"
        
        # Old one is still there, without hash or with None (since it was migrated)
        assert history_after[1]["Filename"] == "old_img.jpg"
        assert "_hash" not in history_after[1]
        
    finally:
        # Clean up
        gc.collect()
        for _ in range(5):
            try:
                os.unlink(path)
                break
            except PermissionError:
                time.sleep(0.1)
