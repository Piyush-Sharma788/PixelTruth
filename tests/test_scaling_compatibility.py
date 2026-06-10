import cv2
import numpy as np
import tensorflow as tf
try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except (ImportError, ModuleNotFoundError):
    from keras.applications.mobilenet_v2 import preprocess_input
try:
    from tensorflow.keras.layers import Rescaling
except (ImportError, ModuleNotFoundError):
    from keras.layers import Rescaling
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


def test_training_vs_inference_preprocessing_pipeline():
    # Generate a fixed random mock image simulating a raw BGR camera frame/file
    np.random.seed(100)
    mock_bgr_image = np.random.randint(0, 256, size=(120, 160, 3)).astype(np.uint8)

    # 1. Inference preprocessing path: BGR array preprocessed to batch of size 1 (RGB, range [0, 255])
    inference_processed = preprocess_image_array(mock_bgr_image)

    # 2. Training-time preprocessing path:
    # Keras load & resize pipelines read image files in RGB format, resizing to (96, 96) in [0.0, 255.0]
    mock_rgb_image = cv2.cvtColor(mock_bgr_image, cv2.COLOR_BGR2RGB)
    training_resized = cv2.resize(mock_rgb_image, (96, 96)).astype(np.float32)
    training_processed = np.expand_dims(training_resized, axis=0)

    # Assert both preprocessing paths yield identical [0.0, 255.0] range arrays
    assert np.allclose(inference_processed, training_processed, atol=1e-5)

    # 3. Model validation check:
    # Model embeds a Rescaling(scale=1./127.5, offset=-1.) layer as its first stage.
    # Verify that feeding both paths' outputs to the model's preprocessing layer(s)
    # produces identical values entering the MobileNetV2 backbone.
    model = tf.keras.models.Sequential([
        Rescaling(scale=1./127.5, offset=-1., input_shape=(96, 96, 3))
    ])

    inference_model_input = model(inference_processed).numpy()
    training_model_input = model(training_processed).numpy()

    # Tensors entering the backbone must match perfectly
    assert np.allclose(inference_model_input, training_model_input, atol=1e-5)

    # They must also match Keras applications' canonical MobileNetV2 preprocess_input contract
    canonical_preprocessed = preprocess_input(training_resized)
    canonical_preprocessed = np.expand_dims(canonical_preprocessed, axis=0)
    assert np.allclose(inference_model_input, canonical_preprocessed, atol=1e-5)

