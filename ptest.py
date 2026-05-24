"""
Unit tests for the PixelTruth deepfake detection pipeline.

Run with:
    pytest ptest.py -v

These tests exercise the **unified pipeline** directly through ``predict.py``,
``preprocessing.py``, and ``config.py``.  There is **no dependency on Streamlit
or app.py** — the tests never import the frontend and never stub
``sys.modules``.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import config
import preprocessing
import predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blank_image(h: int = 200, w: int = 200, channels: int = 3) -> np.ndarray:
    """Return a solid-colour BGR image as a numpy array."""
    return np.full((h, w, channels), 128, dtype=np.uint8)


def make_mock_model(prediction_array: np.ndarray) -> MagicMock:
    """Return a mock Keras model whose .predict() returns prediction_array."""
    mock = MagicMock()
    mock.predict.return_value = prediction_array
    return mock


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a BGR numpy array into raw PNG bytes."""
    ok, encoded = cv2.imencode(".png", image)
    assert ok, "cv2.imencode failed"
    return encoded.tobytes()


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:

    def test_image_size_is_tuple_of_two_ints(self):
        assert isinstance(config.IMAGE_SIZE, tuple)
        assert len(config.IMAGE_SIZE) == 2
        assert all(isinstance(d, int) for d in config.IMAGE_SIZE)

    def test_low_confidence_threshold_is_numeric(self):
        assert isinstance(config.LOW_CONFIDENCE_THRESHOLD, (float, int))

    def test_low_confidence_threshold_in_valid_range(self):
        """Must be strictly between 0.5 and 1.0."""
        t = config.LOW_CONFIDENCE_THRESHOLD
        assert 0.5 < t < 1.0, (
            f"LOW_CONFIDENCE_THRESHOLD={t} is outside the valid range (0.5, 1.0)"
        )

    def test_default_model_path_is_string(self):
        assert isinstance(config.DEFAULT_MODEL_PATH, str)
        assert config.DEFAULT_MODEL_PATH.endswith(".h5")

    def test_supported_extensions_is_frozenset(self):
        assert isinstance(config.SUPPORTED_EXTENSIONS, frozenset)
        assert ".jpg" in config.SUPPORTED_EXTENSIONS
        assert ".png" in config.SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# preprocessing — array path
# ---------------------------------------------------------------------------

class TestPreprocessImageArray:

    def test_output_shape(self):
        """Output tensor must always be (1, 96, 96, 3) regardless of input size."""
        for size in [(50, 50), (200, 200), (1024, 768)]:
            image = make_blank_image(*size)
            result = preprocessing.preprocess_image_array(image)
            assert result.shape == (1, 96, 96, 3), (
                f"Expected (1, 96, 96, 3) for input size {size}, got {result.shape}"
            )

    def test_pixel_values_remain_in_model_input_range(self):
        image = make_blank_image()
        result = preprocessing.preprocess_image_array(image)
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_white_image_remains_255_for_model_rescaling_layer(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = preprocessing.preprocess_image_array(image)
        assert np.allclose(result, 255.0)

    def test_black_image_remains_0(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocessing.preprocess_image_array(image)
        assert np.allclose(result, 0.0)

    def test_output_dtype_is_float(self):
        image = make_blank_image()
        result = preprocessing.preprocess_image_array(image)
        assert np.issubdtype(result.dtype, np.floating)

    def test_bgr_to_rgb_conversion(self):
        """After preprocessing, channel 0 must be the original red value."""
        bgr = np.zeros((96, 96, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10   # blue
        bgr[:, :, 1] = 128  # green
        bgr[:, :, 2] = 200  # red

        result = preprocessing.preprocess_image_array(bgr)

        assert np.allclose(result[0, :, :, 0], 200, atol=1e-5), "BGR to RGB channel 0 mismatch"
        assert np.allclose(result[0, :, :, 2], 10, atol=1e-5), "BGR to RGB channel 2 mismatch"

    def test_none_input_raises(self):
        with pytest.raises(Exception):
            preprocessing.preprocess_image_array(None)


# ---------------------------------------------------------------------------
# preprocessing — bytes path
# ---------------------------------------------------------------------------

class TestPreprocessImageBytes:

    def test_output_shape(self):
        raw = image_to_png_bytes(make_blank_image())
        result = preprocessing.preprocess_image_bytes(raw)
        assert result.shape == (1, 96, 96, 3)

    def test_invalid_bytes_raises(self):
        with pytest.raises(Exception):
            preprocessing.preprocess_image_bytes(b"not an image")

    def test_caching_returns_same_result(self):
        """Identical bytes should hit the LRU cache and return the same object."""
        preprocessing.preprocess_image_bytes.cache_clear()
        raw = image_to_png_bytes(make_blank_image())

        first = preprocessing.preprocess_image_bytes(raw)
        second = preprocessing.preprocess_image_bytes(raw)

        assert first is second, "Cache did not return the same object for identical input"


# ---------------------------------------------------------------------------
# predict.preprocess_image — unified entry point
# ---------------------------------------------------------------------------

class TestUnifiedPreprocessImage:

    def test_accepts_numpy_array(self):
        result = predict.preprocess_image(make_blank_image())
        assert result.shape == (1, 96, 96, 3)

    def test_accepts_bytes(self):
        raw = image_to_png_bytes(make_blank_image())
        result = predict.preprocess_image(raw)
        assert result.shape == (1, 96, 96, 3)

    def test_accepts_file_path(self):
        img = make_blank_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            path = f.name
        try:
            result = predict.preprocess_image(path)
            assert result.shape == (1, 96, 96, 3)
        finally:
            os.unlink(path)

    def test_file_path_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            predict.preprocess_image("/nonexistent/image.png")

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                predict.preprocess_image(path)
        finally:
            os.unlink(path)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            predict.preprocess_image(12345)

    def test_array_and_bytes_produce_identical_output(self):
        """Preprocessing via numpy array or via bytes of that same image must
        produce the exact same tensor."""
        img = make_blank_image(100, 100)
        raw = image_to_png_bytes(img)

        from_array = predict.preprocess_image(img)
        from_bytes = predict.preprocess_image(raw)

        assert np.allclose(from_array, from_bytes, atol=1e-5), (
            "Array-path and bytes-path produce different tensors"
        )


# ---------------------------------------------------------------------------
# predict.predict_image — unified prediction
# ---------------------------------------------------------------------------

class TestPredictImage:

    def test_returns_fake_label_when_class_0_wins(self):
        mock_model = make_mock_model(np.array([[0.9, 0.1]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert result["label"] == "Fake"

    def test_returns_real_label_when_class_1_wins(self):
        mock_model = make_mock_model(np.array([[0.1, 0.9]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert result["label"] == "Real"

    def test_confidence_is_float_between_0_and_1(self):
        mock_model = make_mock_model(np.array([[0.8, 0.2]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_result_contains_expected_keys(self):
        mock_model = make_mock_model(np.array([[0.7, 0.3]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert "label" in result
        assert "confidence" in result
        assert "raw" in result
        assert "processed_image" in result

    def test_model_receives_correct_shape(self):
        mock_model = make_mock_model(np.array([[0.6, 0.4]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            predict.predict_image(make_blank_image(h=300, w=400))
        call_args = mock_model.predict.call_args[0][0]
        assert call_args.shape == (1, 96, 96, 3)

    def test_label_is_one_of_valid_classes(self):
        for pred in [np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]])]:
            mock_model = make_mock_model(pred)
            with patch.object(predict, "load_cached_model", return_value=mock_model):
                result = predict.predict_image(make_blank_image())
            assert result["label"] in ("Real", "Fake")

    def test_path_input_includes_image_key(self):
        """When a file path is provided, the result dict should include 'image'."""
        img = make_blank_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            path = f.name
        try:
            mock_model = make_mock_model(np.array([[0.8, 0.2]]))
            with patch.object(predict, "load_cached_model", return_value=mock_model):
                result = predict.predict_image(path)
            assert "image" in result
            assert result["image"] == path
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# predict.predict_image_tuple — backward-compat wrapper
# ---------------------------------------------------------------------------

class TestPredictImageTuple:

    def test_returns_three_element_tuple(self):
        mock_model = make_mock_model(np.array([[0.9, 0.1]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image_tuple(make_blank_image())
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_none_triple_when_model_fails(self):
        with patch.object(predict, "load_cached_model", side_effect=Exception("no model")):
            label, confidence, processed = predict.predict_image_tuple(make_blank_image())
        assert label is None
        assert confidence is None
        assert processed is None


# ---------------------------------------------------------------------------
# LOW_CONFIDENCE_THRESHOLD logic
# ---------------------------------------------------------------------------

class TestLowConfidenceThreshold:

    def test_low_confidence_real_label_unchanged(self):
        """predict_image must still return 'Real' even when confidence < threshold."""
        mock_model = make_mock_model(np.array([[0.38, 0.62]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert result["label"] == "Real"
        assert abs(result["confidence"] - 0.62) < 1e-5

    def test_low_confidence_fake_label_unchanged(self):
        mock_model = make_mock_model(np.array([[0.55, 0.45]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert result["label"] == "Fake"
        assert abs(result["confidence"] - 0.55) < 1e-5

    def test_is_uncertain_true_below_threshold(self):
        t = config.LOW_CONFIDENCE_THRESHOLD
        for conf in [0.51, 0.60, 0.62, 0.69, t - 0.001]:
            assert conf < t

    def test_is_uncertain_false_at_or_above_threshold(self):
        t = config.LOW_CONFIDENCE_THRESHOLD
        for conf in [t, t + 0.001, 0.80, 0.95, 1.0]:
            assert conf >= t

    def test_high_confidence_real_not_uncertain(self):
        mock_model = make_mock_model(np.array([[0.08, 0.92]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert result["label"] == "Real"
        assert result["confidence"] >= config.LOW_CONFIDENCE_THRESHOLD

    def test_high_confidence_fake_not_uncertain(self):
        mock_model = make_mock_model(np.array([[0.95, 0.05]]))
        with patch.object(predict, "load_cached_model", return_value=mock_model):
            result = predict.predict_image(make_blank_image())
        assert result["label"] == "Fake"
        assert result["confidence"] >= config.LOW_CONFIDENCE_THRESHOLD
