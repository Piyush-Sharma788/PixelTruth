"""
Unit tests for preprocess_image and predict_image in app.py.

Run with:
    pytest test_pipeline.py -v

Imports the real functions directly from app.py.
Streamlit and the metrics module are stubbed out in sys.modules before
the import so that Streamlit's module-level setup does not run during
test collection.  The real model file is not required — tests that
exercise predict_image patch app.model with a mock.
"""

import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import preprocessing

# ---------------------------------------------------------------------------
# Stub Streamlit and metrics before app.py is imported.
# Streamlit runs page-config, column layout, and widget calls at import
# time; mocking the module prevents those from erroring in a headless env.
# ---------------------------------------------------------------------------

_st_mock = MagicMock()
_st_mock.columns.side_effect = lambda spec: [
    MagicMock() for _ in (spec if isinstance(spec, list) else range(spec))
]
# Allow @st.cache_resource to pass the decorated function through unchanged
# so load_deepfake_model remains a real callable (it returns None when the
# .h5 file is absent, which is the correct CI behaviour).
_st_mock.cache_resource = lambda f: f
_st_mock.file_uploader.return_value = None

_metrics_mock = MagicMock()
_metrics_mock.get_sample_metrics.return_value = {
    "accuracy": 95.0,
    "precision": 94.0,
    "recall": 93.0,
    "f1_score": 93.5,
}
_metrics_mock.get_class_statistics.return_value = {}

sys.modules.setdefault("streamlit", _st_mock)
sys.modules.setdefault("streamlit.components", MagicMock())
sys.modules.setdefault("streamlit.components.v1", MagicMock())
sys.modules.setdefault("metrics", _metrics_mock)

import app  # noqa: E402 — must follow sys.modules stubs
from app import predict_image, preprocess_image  # noqa: E402
from app import preprocess_uploaded_image  # noqa: E402


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


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:

    def test_output_shape_is_1_96_96_3(self):
        """Output tensor must always be (1, 96, 96, 3) regardless of input size."""
        for size in [(50, 50), (200, 200), (1024, 768)]:
            image = make_blank_image(*size)
            result = preprocess_image(image)
            assert result.shape == (1, 96, 96, 3), (
                f"Expected (1, 96, 96, 3) for input size {size}, got {result.shape}"
            )

    def test_pixel_values_normalised_between_0_and_1(self):
        """All pixel values must be in [0.0, 1.0] after preprocessing."""
        image = make_blank_image()
        result = preprocess_image(image)
        assert result.min() >= 0.0, "Pixel values dropped below 0.0"
        assert result.max() <= 1.0, "Pixel values exceeded 1.0"

    def test_all_white_image_normalises_to_1(self):
        """A pure-white image (255, 255, 255) should normalise to 1.0."""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = preprocess_image(image)
        assert np.allclose(result, 1.0), "White image did not normalise to 1.0"

    def test_all_black_image_normalises_to_0(self):
        """A pure-black image (0, 0, 0) should normalise to 0.0."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_image(image)
        assert np.allclose(result, 0.0), "Black image did not normalise to 0.0"

    def test_output_dtype_is_float(self):
        """Output must be a float array, not uint8."""
        image = make_blank_image()
        result = preprocess_image(image)
        assert np.issubdtype(result.dtype, np.floating), (
            f"Expected float dtype, got {result.dtype}"
        )

    def test_non_square_input_resized_correctly(self):
        """A non-square image must still produce (1, 96, 96, 3)."""
        image = make_blank_image(h=480, w=640)
        result = preprocess_image(image)
        assert result.shape == (1, 96, 96, 3)

    def test_none_input_raises(self):
        """Passing None should raise an error, not return silently wrong data."""
        with pytest.raises(Exception):
            preprocess_image(None)

    def test_bgr_input_is_converted_to_rgb(self):
        """
        preprocess_image must convert BGR→RGB so the model receives channels
        in the same order as the RGB training pipeline (ImageDataGenerator/PIL).

        Strategy: build a synthetic BGR image where blue, green, and red pixel
        values are all distinct.  After preprocess_image the tensor's channel 0
        must equal the original red value (200/255) and channel 2 must equal
        the original blue value (10/255).  If the BGR→RGB conversion is absent,
        channel 0 would contain blue (10/255) instead — and the assertions fail.
        """
        # Construct a solid-colour BGR image: B=10, G=128, R=200
        bgr_image = np.zeros((96, 96, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 10   # blue  channel (OpenCV channel 0)
        bgr_image[:, :, 1] = 128  # green channel
        bgr_image[:, :, 2] = 200  # red   channel (OpenCV channel 2)

        result = preprocess_image(bgr_image)  # expected shape: (1, 96, 96, 3)

        expected_ch0 = 200 / 255.0  # after BGR→RGB, channel 0 = original red
        expected_ch2 = 10 / 255.0   # after BGR→RGB, channel 2 = original blue

        assert np.allclose(result[0, :, :, 0], expected_ch0, atol=1e-5), (
            f"Channel 0 of output is {result[0, 0, 0, 0]:.4f}, expected {expected_ch0:.4f} "
            "(original red). BGR→RGB conversion appears to be missing."
        )
        assert np.allclose(result[0, :, :, 2], expected_ch2, atol=1e-5), (
            f"Channel 2 of output is {result[0, 0, 0, 2]:.4f}, expected {expected_ch2:.4f} "
            "(original blue). BGR→RGB conversion appears to be missing."
        )


# ---------------------------------------------------------------------------
# predict_image
# ---------------------------------------------------------------------------

class TestPredictImage:

    def test_returns_real_label_when_class_0_wins(self):
        """When model predicts class 0 with high confidence, label must be 'Real'."""
        mock_model = make_mock_model(np.array([[0.9, 0.1]]))
        with patch.object(app, "model", mock_model):
            label, confidence = predict_image(make_blank_image())
        assert label == "Real"

    def test_returns_fake_label_when_class_1_wins(self):
        """When model predicts class 1 with high confidence, label must be 'Fake'."""
        mock_model = make_mock_model(np.array([[0.1, 0.9]]))
        with patch.object(app, "model", mock_model):
            label, confidence = predict_image(make_blank_image())
        assert label == "Fake"

    def test_confidence_is_float(self):
        """Confidence score must be a Python float."""
        mock_model = make_mock_model(np.array([[0.8, 0.2]]))
        with patch.object(app, "model", mock_model):
            label, confidence = predict_image(make_blank_image())
        assert isinstance(confidence, float), (
            f"Expected float confidence, got {type(confidence)}"
        )

    def test_confidence_between_0_and_1(self):
        """Confidence must be in [0.0, 1.0]."""
        mock_model = make_mock_model(np.array([[0.7, 0.3]]))
        with patch.object(app, "model", mock_model):
            label, confidence = predict_image(make_blank_image())
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_none_model_returns_none_none(self):
        """If no model is loaded, both return values must be None."""
        with patch.object(app, "model", None):
            label, confidence = predict_image(make_blank_image())
        assert label is None
        assert confidence is None

    def test_model_receives_preprocessed_input(self):
        """model.predict must be called with a tensor of shape (1, 96, 96, 3)."""
        mock_model = make_mock_model(np.array([[0.6, 0.4]]))
        with patch.object(app, "model", mock_model):
            predict_image(make_blank_image(h=300, w=400))
        call_args = mock_model.predict.call_args[0][0]
        assert call_args.shape == (1, 96, 96, 3), (
            f"model.predict received wrong shape: {call_args.shape}"
        )


    def test_label_is_one_of_valid_classes(self):
        """Label must be exactly 'Real' or 'Fake', nothing else."""
        for prediction in [np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]])]:
            mock_model = make_mock_model(prediction)
            with patch.object(app, "model", mock_model):
                label, _ = predict_image(make_blank_image())
            assert label in ("Real", "Fake"), f"Unexpected label: {label}"

    def test_uploaded_image_preprocessing_is_cached_for_identical_bytes(self):
        """Repeated preprocessing of the same upload should reuse the cached tensor."""
        ok, encoded = cv2.imencode(".png", make_blank_image())
        assert ok
        image_bytes = encoded.tobytes()

        preprocess_uploaded_image.cache_clear()

        with patch.object(preprocessing, "preprocess_image_array", wraps=preprocessing.preprocess_image_array) as wrapped:
            first = preprocess_uploaded_image(image_bytes)
            second = preprocess_uploaded_image(image_bytes)

        assert wrapped.call_count == 1
        assert np.array_equal(first, second)