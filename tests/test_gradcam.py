import os
# Force TensorFlow to use CPU only to avoid GPU-related segmentation faults during testing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Attempt to import TensorFlow; if unavailable, skip all tests in this file
try:
    import tensorflow as tf
except Exception:
    import pytest
    pytest.skip("TensorFlow import failed, skipping GradCAM tests", allow_module_level=True)

import numpy as np
import pytest

from inference import find_last_conv_layer
from gradcam import make_gradcam_heatmap

def test_find_last_conv_layer_nested_backbone():
    # 1. Create a nested backbone Sequential model
    backbone = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(96, 96, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu", name="nested_conv"),
        ]
    )

    # 2. Wrap backbone inside a Sequential classifier model
    model = tf.keras.Sequential(
        [
            backbone,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 3. Verify it recursively finds the convolutional layer
    conv_layer = find_last_conv_layer(model)
    assert isinstance(conv_layer, tf.keras.layers.Layer)
    assert conv_layer.name == "nested_conv"

    # 4. Verify Grad-CAM can be generated from the full model
    img = np.zeros((1, 96, 96, 3), dtype=np.float32)
    heatmap = make_gradcam_heatmap(img, model, conv_layer)
    assert heatmap.shape == (94, 94)  # 96 - 3 + 1 = 94
    assert np.max(heatmap) <= 1.0
    assert np.min(heatmap) >= 0.0


def test_find_last_conv_layer_functional_model():
    # 1. Create a simple Functional model
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Conv2D(4, 3, activation="relu", name="func_conv")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 2. Verify find_last_conv_layer returns the layer object
    conv_layer = find_last_conv_layer(model)
    assert conv_layer.name == "func_conv"

    # 3. Verify Grad-CAM generation with explicit target class index (e.g. 1)
    img = np.zeros((1, 96, 96, 3), dtype=np.float32)
    heatmap = make_gradcam_heatmap(img, model, conv_layer, pred_index=1)
    assert heatmap.shape == (94, 94)
    assert np.max(heatmap) <= 1.0


def test_find_last_conv_layer_functional_backbone_in_sequential_classifier():
    backbone_inputs = tf.keras.Input(shape=(96, 96, 3))
    backbone_outputs = tf.keras.layers.Conv2D(
        4, 3, activation="relu", name="backbone_conv"
    )(backbone_inputs)
    backbone = tf.keras.Model(backbone_inputs, backbone_outputs)
    model = tf.keras.Sequential(
        [
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    conv_layer = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(
        np.zeros((1, 96, 96, 3), dtype=np.float32), model, conv_layer
    )

    assert conv_layer.name == "backbone_conv"
    assert heatmap.shape == (94, 94)


def test_find_last_conv_layer_missing_raises_value_error():
    # 1. Create a model without any convolutional layers
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(96, 96, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 2. Verify find_last_conv_layer raises ValueError
    with pytest.raises(ValueError, match="No convolutional layer found"):
        find_last_conv_layer(model)


def test_make_gradcam_heatmap_fallback_by_name():
    # Verify that passing layer name as string fallback is still supported
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Conv2D(4, 3, activation="relu", name="target_conv_name")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    img = np.zeros((1, 96, 96, 3), dtype=np.float32)
    heatmap = make_gradcam_heatmap(img, model, "target_conv_name")
    assert heatmap.shape == (94, 94)
