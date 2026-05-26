from functools import lru_cache
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import IMAGE_SIZE
from exceptions import PreprocessingError

MIN_IMAGE_DIM = 10


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
        When the bytes cannot be decoded into a valid image.
    """
    # Protect against decompression bombs by checking dimensions first
    MAX_IMAGE_PIXELS = 50_000_000  # conservative cap (~7k x 7k)

    try:
        with Image.open(BytesIO(image_bytes)) as pil_img:
            w, h = pil_img.size
            if (w * h) > MAX_IMAGE_PIXELS:
                raise PreprocessingError(
                    "Image exceeds maximum allowed pixel area and may be unsafe to process."
                )
    except UnidentifiedImageError:
        raise PreprocessingError("The uploaded file is not a recognizable image format.")
    except PreprocessingError:
        raise
    except Exception:
        # Any other PIL errors are surfaced as preprocessing issues
        raise PreprocessingError("Failed to validate image bytes before decoding.")

    file_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError(
            "The uploaded file appears to be corrupted or is not a valid image."
        )
    return image


@lru_cache(maxsize=32)
def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode *and* preprocess raw image bytes in one shot."""
    return preprocess_image_array(decode_image_bytes(image_bytes))
