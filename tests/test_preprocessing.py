import numpy as np
import pytest

from preprocessing import batch_preprocess, preprocess_image_array, preprocess_image_bytes


def test_preprocess_image_array_resizes_and_converts_bgr_to_rgb():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    result = preprocess_image_array(image)

    assert result.shape == (1, 96, 96, 3)
    assert result.dtype == np.float32
    assert result[0, 0, 0].tolist() == [0.0, 0.0, 255.0]


def test_preprocess_image_array_accepts_grayscale_and_bgra_images():
    grayscale = np.zeros((12, 12), dtype=np.uint8)
    bgra = np.zeros((12, 12, 4), dtype=np.uint8)

    assert preprocess_image_array(grayscale).shape == (1, 96, 96, 3)
    assert preprocess_image_array(bgra).shape == (1, 96, 96, 3)


def test_batch_preprocess_validates_input():
    with pytest.raises(ValueError, match="empty"):
        batch_preprocess([])

    with pytest.raises(ValueError, match="too small"):
        preprocess_image_array(np.zeros((5, 5, 3), dtype=np.uint8))


def test_uploaded_bytes_are_not_retained_in_cache():
    import cv2

    ok, encoded = cv2.imencode(".png", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok

    preprocess_image_bytes.cache_clear()
    preprocess_image_bytes(encoded.tobytes())

    assert preprocess_image_bytes.cache_info().currsize == 0
