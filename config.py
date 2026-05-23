"""
Central configuration for the PixelTruth deepfake detection project.

All magic numbers, environment variable keys, and shared constants live
here so that every module can import them from a single source of truth.
"""

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

IMAGE_SIZE: tuple[int, int] = (96, 96)
"""Target (width, height) to which every input image is resized before
being fed to the model."""

# ---------------------------------------------------------------------------
# Prediction confidence
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD: float = 0.70
"""Predictions whose winning softmax probability is below this value are
shown as "Low Confidence — Uncertain" instead of a firm Real/Fake verdict.
Raise or lower this value to widen or narrow the uncertain band."""

# ---------------------------------------------------------------------------
# Model paths & environment variable keys
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH: str = "deepfake_detection_model.h5"
"""Default filename/path for the Keras model when no env var is set."""

MODEL_PATH_ENV: str = "PIXELTRUTH_MODEL_PATH"
"""Environment variable that overrides the default model file path."""

MODEL_URL_ENV: str = "PIXELTRUTH_MODEL_URL"
"""Environment variable pointing to a download URL for the model file."""

MODEL_SHA256_ENV: str = "PIXELTRUTH_MODEL_SHA256"
"""Environment variable containing the expected SHA-256 hex digest
of the model file (used for integrity verification)."""

# ---------------------------------------------------------------------------
# Supported image extensions (used by CLI predict.py)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
"""Default log-line format used across modules."""
