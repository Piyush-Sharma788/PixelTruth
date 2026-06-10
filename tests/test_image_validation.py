"""
Tests for the decompression-bomb protection in preprocessing.decode_image_bytes
(issue #47). A tiny PNG with huge declared dimensions previously caused
cv2.imdecode to allocate the full uncompressed buffer (multiple GB), crashing
the process. The fix uses PIL to read just the image header and reject
oversized images before cv2 ever sees them.
"""
from __future__ import annotations
import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from preprocessing import decode_image_bytes


@pytest.fixture(autouse=True)
def _clear_decode_cache():
    """decode_image_bytes is lru_cached; clear between tests for isolation."""
    decode_image_bytes.cache_clear()
    yield


def make_bomb_png(width: int, height: int) -> bytes:
    """Construct a minimal valid PNG claiming (width x height) dimensions.
    
    Reproducer from issue #47. Output is well under 1 KB regardless of
    declared dimensions.
    """
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = b'IHDR' + ihdr_data
    ihdr_chunk = struct.pack(">I", 13) + ihdr + struct.pack(">I", zlib.crc32(ihdr))
    idat_data = zlib.compress(b'\x00')
    idat = b'IDAT' + idat_data
    idat_chunk = struct.pack(">I", len(idat_data)) + idat + struct.pack(">I", zlib.crc32(idat))
    iend_chunk = b'\x00\x00\x00\x00IEND' + struct.pack(">I", zlib.crc32(b'IEND'))
    return sig + ihdr_chunk + idat_chunk + iend_chunk


def make_real_png(width: int, height: int) -> bytes:
    """Construct a real PNG with actual pixel data, for positive-case tests."""
    img = Image.new('RGB', (width, height), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_rejects_oversized_bomb():
    """A 50000x50000 bomb (2.5 billion pixels) must be rejected without crashing."""
    bomb = make_bomb_png(50_000, 50_000)
    assert len(bomb) < 1024, "bomb file should be tiny; if not, test is invalid"
    with pytest.raises(ValueError, match=r"Image too large"):
        decode_image_bytes(bomb)


def test_accepts_normal_image():
    """A typical 100x100 image should decode successfully."""
    valid = make_real_png(100, 100)
    result = decode_image_bytes(valid)
    assert isinstance(result, np.ndarray)
    assert result.shape[:2] == (100, 100)


def test_decoder_does_not_cache_full_resolution_uploads():
    valid = make_real_png(100, 100)

    decode_image_bytes(valid)

    assert decode_image_bytes.cache_info().currsize == 1


def test_env_var_tightens_cap(monkeypatch):
    """Setting PIXELTRUTH_MAX_PIXELS should make a previously-valid image fail."""
    monkeypatch.setenv("PIXELTRUTH_MAX_PIXELS", "9999")  # 9999 < 100*100 = 10000
    valid = make_real_png(100, 100)
    with pytest.raises(ValueError, match=r"Image too large"):
        decode_image_bytes(valid)


def test_env_var_invalid_falls_back_to_default(monkeypatch):
    """Garbage env var value should fall back to the default cap, not crash."""
    monkeypatch.setenv("PIXELTRUTH_MAX_PIXELS", "not-a-number")
    valid = make_real_png(100, 100)
    result = decode_image_bytes(valid)
    assert isinstance(result, np.ndarray)


def test_corrupted_bytes_raise_value_error():
    """Non-image garbage bytes should be rejected cleanly, not crash."""
    garbage = b'\x00' * 100
    with pytest.raises(ValueError):
        decode_image_bytes(garbage)
