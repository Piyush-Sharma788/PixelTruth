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
Raise or lower this value to widen or narrow the uncertain band.

For sigmoid models (single output) the prediction is used directly.
For softmax models (2 outputs) the max probability is used."""

DEFAULT_TEMPERATURE: float = 1.4
"""Default confidence-calibration temperature shared by the Streamlit app,
FastAPI and CLI so the same image gets the same confidence everywhere.
1.4 is the value fitted (minimum NLL) on the 10k-image validation split by
experiments/train_v4.py for the bundled model (see
deepfake_detection_model.json -> calibration_temperature).  1.0 would
report the raw, slightly over-confident sigmoid output.  Temperature never
changes the Real/Fake verdict, only the confidence."""

MODEL_VERSION: str = "v2.0.0"
"""Model input contract version.  v2 = Haar face crop (full-image fallback)
-> RGB -> 96x96 float32 in [0, 255]; the model rescales internally.
Implemented by ``preprocessing.prepare_model_input``."""

MODEL_OUTPUT_ACTIVATION: str = "sigmoid"
"""Activation function of the model's output layer: 'sigmoid' or 'softmax'.
The evaluation and prediction modules use this to interpret model outputs.
Default is 'sigmoid' to match the three training scripts."""

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
