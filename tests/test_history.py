import os
import sqlite3
import tempfile
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
    
    # Clean up after tests
    os.unlink(path)

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
        filename="hash_test.jpg",
        verdict="Fake",
        confidence_pct=88.2,
        face_detected=0,
        file_hash="abc123hashvalue",
        db_path=temp_db
    )
    
    history = load_history(db_path=temp_db)
    assert len(history) == 1
    assert history[0]["Filename"] == "hash_test.jpg"
    assert history[0]["Result"] == "Fake"
    assert history[0]["Confidence (%)"] == "88.2"
    assert history[0]["Face Detected"] is False
    assert history[0]["_hash"] == "abc123hashvalue"
