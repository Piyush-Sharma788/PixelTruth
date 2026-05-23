from functools import lru_cache

import cv2
import numpy as np
from pathlib import Path

from config import IMAGE_SIZE

MIN_IMAGE_DIM = 10

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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    image = cv2.resize(image, TARGET_IMAGE_SIZE)
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
        When the bytes cannot be decoded into a valid image.
    """
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