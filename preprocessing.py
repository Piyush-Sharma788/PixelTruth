from functools import lru_cache
from pathlib import Path
from io import BytesIO
import io
import os
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import IMAGE_SIZE

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

MIN_IMAGE_DIM = 10

MAX_PIXELS_ENV = "PIXELTRUTH_MAX_PIXELS"
DEFAULT_MAX_PIXELS = 25_000_000


def _get_max_pixels() -> int:
    """Read the pixel cap from the env var, fall back to the default."""
    raw = os.getenv(MAX_PIXELS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_PIXELS
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_MAX_PIXELS
    except ValueError:
        return DEFAULT_MAX_PIXELS


def _validate_compressed_image_dimensions(image_bytes: bytes) -> None:
    """Reject image bytes whose declared dimensions exceed the configured cap."""
    max_pixels = _get_max_pixels()
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except Image.DecompressionBombError as exc:
        raise ValueError(
            f"Image too large: declared dimensions exceed safety limits "
            f"(maximum is {max_pixels:,} pixels)."
        ) from exc
    except Exception as exc:
        raise ValueError(
            "The uploaded file appears to be corrupted or is not a valid image."
        ) from exc

    pixel_count = width * height
    if pixel_count > max_pixels:
        raise ValueError(
            f"Image too large: {width}x{height} ({pixel_count:,} pixels). "
            f"Maximum is {max_pixels:,} pixels. Try resizing the image first."
        )


def validate_image_dimensions(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Image must be a two- or three-dimensional numpy array.")

    h, w = image.shape[:2]
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise ValueError(
            f"Image too small ({w}x{h} px). "
            f"Minimum is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM} px."
        )


def preprocess_image_array(image: np.ndarray) -> np.ndarray:
    validate_image_dimensions(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image channel count: {image.shape[2]}.")

    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image_from_path(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at: {path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image at '{path}'.")

    return preprocess_image_array(image)


def get_image_metadata(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {"height": h, "width": w, "channels": channels}


def batch_preprocess(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("Received an empty list.")
    return np.concatenate([preprocess_image_array(img) for img in images], axis=0)


# Decoded-image caches are deliberately tiny: the cache key is the full raw
# upload (up to 10 MB) and the value can be a 25 MP array (~75 MB).  A small
# bound still lets one request reuse a decode across the predictor and the
# forensic filters without letting a long-running API process grow without
# limit.  Cached arrays are made read-only so no caller can corrupt them.
_DECODE_CACHE_SIZE = 4


@lru_cache(maxsize=_DECODE_CACHE_SIZE)
def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a correctly oriented BGR numpy array (read-only)."""
    _validate_compressed_image_dimensions(image_bytes)

    try:
        pil_image = Image.open(BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.setflags(write=False)
        return image

    except Exception as exc:
        raise ValueError(
            "The uploaded file appears to be corrupted or is not a valid image."
        ) from exc


@lru_cache(maxsize=_DECODE_CACHE_SIZE)
def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode bytes and run the canonical model-input pipeline (read-only result).

    Uses :func:`prepare_model_input`, so the tensor matches what
    ``predict.predict_image`` feeds the model (face crop included).
    """
    batch, _, _ = prepare_model_input(decode_image_bytes(image_bytes))
    batch.setflags(write=False)
    return batch


def get_metadata_from_bytes(image_bytes: bytes) -> dict:
    """Return metadata for an image provided as raw bytes.

    Decodes the bytes and returns height, width, and channel count
    without running the full preprocessing pipeline.

    Args:
        image_bytes: Raw bytes of an encoded image file.

    Returns:
        dict with keys 'height', 'width', and 'channels'.
    """
    image = decode_image_bytes(image_bytes)
    return get_image_metadata(image)


_cascade_local = threading.local()


def _get_face_cascade():
    """Return a per-thread Haar cascade (loaded once per thread).

    ``CascadeClassifier.detectMultiScale`` is not guaranteed thread-safe and
    the FastAPI app runs inference in a thread pool, so each thread keeps its
    own instance instead of re-reading the XML on every call.
    """
    cascade = getattr(_cascade_local, "cascade", None)
    if cascade is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            return None
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None
        _cascade_local.cascade = cascade
    return cascade


def detect_and_crop_face(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Detect the primary face in a BGR image and crop it.

    If no face is found, returns the original image and None.
    The primary face is defined as the face with the largest area (width * height).
    """
    face_cascade = _get_face_cascade()
    if face_cascade is None:
        return image, None

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return image, None

    # Select the primary face (largest by bounding box area)
    primary_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = primary_face

    cropped = image[int(y):int(y+h), int(x):int(x+w)]
    return cropped, (int(x), int(y), int(w), int(h))


def prepare_model_input(
    bgr_image: np.ndarray,
    face_detector=None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
    """Canonical image -> model-input pipeline (model contract v2).

    Training (``scripts/prepare_dataset.py``), evaluation (``evaluate.py``)
    and every inference path (Streamlit, FastAPI, CLI) call this function so
    the tensor the model sees is produced the same way everywhere:

    1. detect the largest frontal face with the OpenCV Haar cascade and crop
       it; if no face is found (or detection errors), use the full image;
    2. convert to RGB, resize to ``IMAGE_SIZE`` and cast to float32 in
       ``[0, 255]`` (the model rescales internally).

    Returns ``(batch, face_image, face_box)`` where ``batch`` has shape
    ``(1, H, W, 3)``, ``face_image`` is the BGR region that was analysed and
    ``face_box`` is ``(x, y, w, h)`` or ``None``.
    """
    detector = face_detector or detect_and_crop_face
    try:
        face_image, face_box = detector(bgr_image)
        return preprocess_image_array(face_image), face_image, face_box
    except Exception:
        # Face detection is best-effort; fall back to the full frame.
        return preprocess_image_array(bgr_image), bgr_image, None
