"""Lightweight shared inference utilities for PixelTruth.

This module avoids heavy import-time side effects by performing
TensorFlow / Keras imports lazily inside functions. It provides a
clean API for preprocessing, prediction, model loading, and a helper
to locate the last convolutional layer name in a model.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from preprocessing import preprocess_image_array, preprocess_image_bytes
from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256
from exceptions import PreprocessingError, ModelExecutionError


def decode_prediction(prediction: np.ndarray) -> Tuple[str, float]:
    """Convert model output to a label/confidence using the canonical decoder."""
    # Local import prevents a module-level dependency cycle.
    from predict import decode_prediction as canonical_decode_prediction

    label, confidence, _ = canonical_decode_prediction(prediction)
    return label, confidence


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image must be a numpy array")
    return preprocess_image_array(image)


def preprocess_uploaded_image(image_bytes: bytes) -> np.ndarray:
    return preprocess_image_bytes(image_bytes)


def load_model_safe(
    model_path: Optional[str] = None,
    model_url: Optional[str] = None,
    model_sha256: Optional[str] = None,
    download_if_missing: bool = True,
):
    resolved_model_path = model_path or get_model_path()
    resolved_model_url = model_url or get_model_url()
    resolved_model_sha256 = model_sha256 or get_model_sha256()

    model_file_path = ensure_model_file(
        model_path=resolved_model_path,
        model_url=resolved_model_url,
        model_sha256=resolved_model_sha256,
        download_if_missing=download_if_missing,
    )

    try:
        # Local import to avoid TensorFlow side-effects during module import
        from tensorflow.keras.models import load_model

        model = load_model(model_file_path)
        return model
    except Exception as exc:  # pragma: no cover - integration behaviour
        # Let callers handle or log the error; wrap in ModelExecutionError
        raise ModelExecutionError(f"Failed to load model: {exc}") from exc


def predict_image(model, image: np.ndarray) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
    """Run prediction on a pre- or unprocessed image.

    If `model` is None, return (None, None, None) so callers can handle
    the missing-model case without exceptions.
    """
    if model is None:
        return None, None, None

    try:
        processed = preprocess_image(image)
    except Exception as exc:
        raise PreprocessingError(f"Preprocessing failed: {exc}") from exc

    try:
        prediction = model.predict(processed, verbose=0)
        label, confidence = decode_prediction(prediction)
        return label, confidence, processed
    except ModelExecutionError:
        raise
    except Exception as exc:
        raise ModelExecutionError(f"Model prediction failed: {exc}") from exc

def predict_image_from_bytes(model, image_bytes: bytes) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
    """Run prediction directly from raw image bytes.

    Combines preprocess_uploaded_image and predict_image in one call
    so callers don't have to chain them manually.

    Args:
        model: Loaded Keras model or None.
        image_bytes: Raw bytes of an encoded image file.

    Returns:
        Tuple of (label, confidence, processed_image), or
        (None, None, None) if model is None.
    """
    if model is None:
        return None, None, None

    try:
        processed = preprocess_uploaded_image(image_bytes)
    except Exception as exc:
        raise PreprocessingError(f"Preprocessing failed: {exc}") from exc

    try:
        prediction = model.predict(processed, verbose=0)
        label, confidence = decode_prediction(prediction)
        return label, confidence, processed
    except ModelExecutionError:
        raise
    except Exception as exc:
        raise ModelExecutionError(f"Model prediction failed: {exc}") from exc
    
def validate_model_output_shape(prediction: np.ndarray) -> None:
    """Validate that model output shape is supported before decoding.

    Args:
        prediction: Raw output array from model.predict().

    Raises:
        ModelExecutionError: If output shape is not 1 or 2 classes.
    """
    scores = np.asarray(prediction).reshape(-1)
    if scores.size not in (1, 2):
        raise ModelExecutionError(
            f"Unsupported model output shape: {np.asarray(prediction).shape}. "
            f"Expected 1 (sigmoid) or 2 (softmax) outputs."
        )