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
    """Convert sigmoid or two-class softmax output to a label and confidence."""
    scores = np.asarray(prediction, dtype=float).reshape(-1)
    if scores.size == 1:
        real_probability = float(scores[0])
        if not 0.0 <= real_probability <= 1.0:
            raise ModelExecutionError("Model returned a probability outside [0, 1].")
        # Training directories are alphabetic: class 0 = fake, class 1 = real.
        if real_probability >= 0.5:
            return "Real", real_probability
        return "Fake", 1.0 - real_probability

    if scores.size == 2:
        class_label = int(np.argmax(scores))
        return ("Real" if class_label == 1 else "Fake"), float(scores[class_label])

    raise ModelExecutionError(
        f"Unsupported model output shape: {np.asarray(prediction).shape}."
    )


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


def find_last_conv_layer(model):
    """Recursively search for the last convolutional layer object in the model."""
    # We want to traverse layers in reverse order.
    # If a layer has a 'layers' attribute, it's a nested container (Sequential or Functional Model).
    # We should search inside it recursively.
    layers = getattr(model, "layers", None) or []
    for layer in reversed(layers):
        if hasattr(layer, "layers") and getattr(layer, "layers"):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                # If no conv layer was found in this sub-model, continue searching other layers
                continue

        clsname = layer.__class__.__name__
        if "Conv" in clsname:
            return layer

    # Fallback using _flatten_layers if available
    try:
        for layer in reversed(list(model._flatten_layers())):
            clsname = layer.__class__.__name__
            if "Conv" in clsname:
                return layer
    except Exception:
        pass

    raise ValueError("No convolutional layer found in the provided model")


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