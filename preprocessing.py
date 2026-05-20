from functools import lru_cache

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


TARGET_IMAGE_SIZE = (96, 96)


def preprocess_image_array(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, TARGET_IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


@lru_cache(maxsize=32)
def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    file_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "The uploaded file appears to be corrupted or is not a valid image."
        )
    return image


@lru_cache(maxsize=32)
def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    return preprocess_image_array(decode_image_bytes(image_bytes))