from functools import lru_cache

import io
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from config import IMAGE_SIZE

MIN_IMAGE_DIM = 10

# --- Decompression bomb protection (issue #47) ---
# cv2.imdecode allocates the full uncompressed pixel buffer regardless of
# how small the compressed file is, so a tiny PNG with huge declared
# dimensions can crash the process with OOM. We use PIL to parse only the
# header (microseconds, no pixel allocation) and reject oversized images
# before cv2 sees them.
MAX_PIXELS_ENV = "PIXELTRUTH_MAX_PIXELS"
DEFAULT_MAX_PIXELS = 25_000_000  # 25 megapixels covers typical phone/DSLR photos


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
    """Reject image bytes whose declared dimensions exceed the configured cap.

    PIL's Image.open only reads the image header; it does not allocate the
    full pixel buffer, so this is safe to run on untrusted input. PIL also
    raises DecompressionBombError on its own (default ~89 MP); we catch that
    and surface our own "too large" message for consistency.
    """
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
        raise ValueError(f"Could not read image header: {exc}") from exc

    pixel_count = width * height
    if pixel_count > max_pixels:
        raise ValueError(
            f"Image too large: {width}x{height} ({pixel_count:,} pixels). "
            f"Maximum is {max_pixels:,} pixels. Try resizing the image first."
        )

def validate_image_dimensions(image: np.ndarray) -> None:
    h, w = image.shape[:2]
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise ValueError(
            f"Image too small ({w}x{h} px). "
            f"Minimum is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM} px."
        )


def preprocess_image_array(image: np.ndarray) -> np.ndarray:
    validate_image_dimensions(image)
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def preprocess_image_from_path(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at: {path}")
    
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

@lru_cache(maxsize=32)
def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR numpy array.

    Raises
    ------
    ValueError
        When the bytes cannot be decoded into a valid image, or when the
        image's declared dimensions exceed the configured pixel cap
        (PIXELTRUTH_MAX_PIXELS env var, default 25 megapixels).
    """
    _validate_compressed_image_dimensions(image_bytes)
    
    file_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "The uploaded file appears to be corrupted or is not a valid image."
        )
    return image


@lru_cache(maxsize=32)
def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode *and* preprocess raw image bytes in one shot."""
    return preprocess_image_array(decode_image_bytes(image_bytes))

