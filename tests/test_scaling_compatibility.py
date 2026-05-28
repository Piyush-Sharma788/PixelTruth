import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Rescaling
from preprocessing import preprocess_image_array


def test_rescaling_layer_matches_mobilenetv2_preprocess_input():
    # Generate a random raw mock image array with values in [0, 255]
    np.random.seed(42)
    mock_image = np.random.randint(0, 256, size=(96, 96, 3)).astype(np.uint8)

    # 1. Training/Inference time raw representation (float32, range [0, 255], RGB order)
    # preprocess_image_array converts BGR to RGB and returns float32 in range [0, 255]
    bgr_image = mock_image.copy()  # Let's treat mock_image as BGR
    preprocessed_raw = preprocess_image_array(bgr_image)  # shape (1, 96, 96, 3), range [0, 255], RGB

    # 2. Canonical preprocess_input contract:
    # Under the original train.py, ImageDataGenerator(preprocessing_function=preprocess_input)
    # would receive the image in RGB shape (96, 96, 3), convert it using preprocess_input, and then batch it.
    rgb_image = preprocessed_raw[0].copy()  # preprocess_image_array converts BGR to RGB
    expected_tensor = preprocess_input(rgb_image)
    expected_tensor = np.expand_dims(expected_tensor, axis=0)  # shape (1, 96, 96, 3)

    # 3. New contract: model has Rescaling(scale=1./127.5, offset=-1.) layer first
    rescaling_layer = Rescaling(scale=1./127.5, offset=-1., input_shape=(96, 96, 3))

    # Pass the raw [0, 255] preprocessed image through the Rescaling layer
    actual_tensor = rescaling_layer(preprocessed_raw).numpy()

    # Verify that the two tensors are numerically identical (within float32 precision)
    assert np.allclose(actual_tensor, expected_tensor, atol=1e-5)

    # Verify values are in [-1, 1]
    assert np.min(actual_tensor) >= -1.0
    assert np.max(actual_tensor) <= 1.0
