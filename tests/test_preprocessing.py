import os
import numpy as np
import pytest

from preprocessing import (
    batch_preprocess,
    decode_image_bytes,
    preprocess_image_array,
    preprocess_image_bytes,
    DECODE_CACHE_SIZE,
    _DEFAULT_DECODE_CACHE_SIZE,
    _get_decode_cache_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(h: int = 20, w: int = 20) -> bytes:
    """Return minimal PNG-encoded bytes for a solid black image."""
    import cv2
    ok, encoded = cv2.imencode(".png", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok, "cv2.imencode failed"
    return encoded.tobytes()


def _make_jpeg_bytes(h: int = 20, w: int = 20) -> bytes:
    """Return minimal JPEG-encoded bytes for a solid black image."""
    import cv2
    ok, encoded = cv2.imencode(".jpg", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok, "cv2.imencode failed"
    return encoded.tobytes()


# ---------------------------------------------------------------------------
# Existing correctness tests (unchanged)
# ---------------------------------------------------------------------------

def test_preprocess_image_array_resizes_and_converts_bgr_to_rgb():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    result = preprocess_image_array(image)

    assert result.shape == (1, 96, 96, 3)
    assert result.dtype == np.float32
    assert result[0, 0, 0].tolist() == [0.0, 0.0, 255.0]


def test_preprocess_image_array_accepts_grayscale_and_bgra_images():
    grayscale = np.zeros((12, 12), dtype=np.uint8)
    bgra = np.zeros((12, 12, 4), dtype=np.uint8)

    assert preprocess_image_array(grayscale).shape == (1, 96, 96, 3)
    assert preprocess_image_array(bgra).shape == (1, 96, 96, 3)


def test_batch_preprocess_validates_input():
    with pytest.raises(ValueError, match="empty"):
        batch_preprocess([])

    with pytest.raises(ValueError, match="too small"):
        preprocess_image_array(np.zeros((5, 5, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# NEW: cache behaviour tests
# ---------------------------------------------------------------------------

class TestDecodeCacheSize:
    """_get_decode_cache_size reads the env var correctly."""

    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PIXELTRUTH_DECODE_CACHE_SIZE", raising=False)
        assert _get_decode_cache_size() == _DEFAULT_DECODE_CACHE_SIZE

    def test_positive_integer_env_var(self, monkeypatch):
        monkeypatch.setenv("PIXELTRUTH_DECODE_CACHE_SIZE", "16")
        assert _get_decode_cache_size() == 16

    def test_zero_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("PIXELTRUTH_DECODE_CACHE_SIZE", "0")
        assert _get_decode_cache_size() == _DEFAULT_DECODE_CACHE_SIZE

    def test_negative_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("PIXELTRUTH_DECODE_CACHE_SIZE", "-4")
        assert _get_decode_cache_size() == _DEFAULT_DECODE_CACHE_SIZE

    def test_non_integer_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("PIXELTRUTH_DECODE_CACHE_SIZE", "banana")
        assert _get_decode_cache_size() == _DEFAULT_DECODE_CACHE_SIZE


class TestDecodeCacheFunctionality:
    """decode_image_bytes and preprocess_image_bytes cache hits/misses."""

    def setup_method(self):
        decode_image_bytes.cache_clear()
        preprocess_image_bytes.cache_clear()

    def teardown_method(self):
        decode_image_bytes.cache_clear()
        preprocess_image_bytes.cache_clear()

    # -- DECODE_CACHE_SIZE is > 0 -------------------------------------------

    def test_module_constant_is_positive(self):
        """DECODE_CACHE_SIZE must be > 0; maxsize=0 disables caching entirely."""
        assert DECODE_CACHE_SIZE > 0, (
            "DECODE_CACHE_SIZE is 0 — lru_cache(maxsize=0) stores nothing. "
            "Set it to a positive integer or None."
        )

    def test_decode_image_bytes_maxsize_is_positive(self):
        """The lru_cache on decode_image_bytes must have maxsize > 0."""
        info = decode_image_bytes.cache_info()
        assert info.maxsize is None or info.maxsize > 0, (
            f"decode_image_bytes cache maxsize={info.maxsize}; "
            "maxsize=0 disables caching entirely."
        )

    def test_preprocess_image_bytes_maxsize_is_positive(self):
        """The lru_cache on preprocess_image_bytes must have maxsize > 0."""
        info = preprocess_image_bytes.cache_info()
        assert info.maxsize is None or info.maxsize > 0, (
            f"preprocess_image_bytes cache maxsize={info.maxsize}; "
            "maxsize=0 disables caching entirely."
        )

    # -- Cache hit on repeated call -----------------------------------------

    def test_decode_image_bytes_caches_repeated_call(self):
        """Second call with the same bytes must produce a cache hit."""
        image_bytes = _make_png_bytes()

        decode_image_bytes(image_bytes)
        info_after_first = decode_image_bytes.cache_info()

        decode_image_bytes(image_bytes)
        info_after_second = decode_image_bytes.cache_info()

        assert info_after_second.hits >= info_after_first.hits + 1, (
            "Expected at least one cache hit on the second identical call to "
            "decode_image_bytes, but hits did not increase."
        )

    def test_preprocess_image_bytes_caches_repeated_call(self):
        """Second call with the same bytes must produce a cache hit."""
        image_bytes = _make_png_bytes()

        preprocess_image_bytes(image_bytes)
        info_after_first = preprocess_image_bytes.cache_info()

        preprocess_image_bytes(image_bytes)
        info_after_second = preprocess_image_bytes.cache_info()

        assert info_after_second.hits >= info_after_first.hits + 1, (
            "Expected at least one cache hit on the second identical call to "
            "preprocess_image_bytes, but hits did not increase."
        )

    # -- Different bytes produces a miss ------------------------------------

    def test_decode_image_bytes_miss_on_different_bytes(self):
        """Two distinct byte strings must each count as a miss."""
        bytes_a = _make_png_bytes(20, 20)
        bytes_b = _make_png_bytes(22, 22)

        decode_image_bytes(bytes_a)
        misses_after_a = decode_image_bytes.cache_info().misses

        decode_image_bytes(bytes_b)
        misses_after_b = decode_image_bytes.cache_info().misses

        assert misses_after_b == misses_after_a + 1

    # -- Output correctness is unchanged ------------------------------------

    def test_decode_image_bytes_returns_bgr_array(self):
        result = decode_image_bytes(_make_png_bytes())
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_preprocess_image_bytes_returns_float32_batch(self):
        result = preprocess_image_bytes(_make_png_bytes())
        assert result.shape == (1, 96, 96, 3)
        assert result.dtype == np.float32

    # -- JPEG input also caches correctly -----------------------------------

    def test_decode_image_bytes_caches_jpeg_input(self):
        image_bytes = _make_jpeg_bytes()

        decode_image_bytes(image_bytes)
        hits_before = decode_image_bytes.cache_info().hits

        decode_image_bytes(image_bytes)
        hits_after = decode_image_bytes.cache_info().hits

        assert hits_after >= hits_before + 1

    # -- Regression: old maxsize=0 behaviour asserted to be ABSENT ----------

    def test_currsize_is_nonzero_after_first_call(self):
        """With maxsize=0 the currsize stays 0 forever — that must not happen."""
        image_bytes = _make_png_bytes()
        decode_image_bytes(image_bytes)
        info = decode_image_bytes.cache_info()
        # maxsize=None means unbounded; maxsize>0 means bounded but non-zero
        if info.maxsize is not None:
            assert info.currsize > 0, (
                "currsize is 0 after a call — this indicates maxsize=0 "
                "which silently disables caching."
            )