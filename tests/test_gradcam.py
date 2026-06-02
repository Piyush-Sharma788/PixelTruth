import pytest
import numpy as np
import tensorflow as tf

from inference import find_last_conv_layer
from gradcam import get_backbone_submodel, make_gradcam_heatmap


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

# ---- Tests for get_backbone_submodel() ----


def test_get_backbone_submodel_returns_nested_model():
    """Pass 1: finds a nested sub-model that contains Conv layers."""
    backbone = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(96, 96, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
        ],
        name="xception_backbone",
    )
    model = tf.keras.Sequential(
        [
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    result = get_backbone_submodel(model)
    assert result is backbone


def test_get_backbone_submodel_functional_backbone():
    """Pass 1: works when the backbone is a Functional (non-Sequential) model."""
    inp = tf.keras.Input(shape=(96, 96, 3))
    out = tf.keras.layers.Conv2D(4, 3, activation="relu")(inp)
    backbone = tf.keras.Model(inp, out, name="func_backbone")

    model = tf.keras.Sequential(
        [
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    result = get_backbone_submodel(model)
    assert result is backbone
    assert result.name == "func_backbone"


def test_get_backbone_submodel_skips_preprocessing_layer():
    """Pass 1: prefers the conv-bearing backbone even when a non-conv
    sub-model (e.g. a preprocessing wrapper) appears earlier in layers."""
    # Preprocessing sub-model with NO conv layers
    prep_inp = tf.keras.Input(shape=(96, 96, 3))
    prep_out = tf.keras.layers.Rescaling(1.0 / 255)(prep_inp)
    preprocess_model = tf.keras.Model(prep_inp, prep_out, name="preprocessor")

    # Backbone sub-model WITH conv layers
    bb_inp = tf.keras.Input(shape=(96, 96, 3))
    bb_out = tf.keras.layers.Conv2D(8, 3, activation="relu")(bb_inp)
    backbone = tf.keras.Model(bb_inp, bb_out, name="real_backbone")

    # Top-level: preprocessor first, then backbone
    top_inp = tf.keras.Input(shape=(96, 96, 3))
    x = preprocess_model(top_inp)
    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(top_inp, outputs)

    result = get_backbone_submodel(model)
    # Should pick the backbone (has Conv), NOT the preprocessor
    assert result is backbone
    assert result.name == "real_backbone"


def test_get_backbone_submodel_flat_model_fallback():
    """Pass 3: for flat models with direct Conv layers and no nesting,
    returns the model itself so Grad-CAM can still work."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(96, 96, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    result = get_backbone_submodel(model)
    assert result is model


def test_get_backbone_submodel_no_conv_raises_with_layer_types():
    """Pass 4: raises ValueError listing actual layer types for debugging."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(96, 96, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    with pytest.raises(ValueError, match="Layer types present"):
        get_backbone_submodel(model)


def test_gradcam_end_to_end_pipeline():
    """Integration: get_backbone_submodel → find_last_conv_layer →
    make_gradcam_heatmap → overlay_heatmap produces a valid image."""
    from gradcam import overlay_heatmap

    # Build a realistic nested architecture
    backbone = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(96, 96, 3)),
            tf.keras.layers.Conv2D(8, 3, activation="relu", name="e2e_conv1"),
            tf.keras.layers.Conv2D(16, 3, activation="relu", name="e2e_conv2"),
        ],
        name="e2e_backbone",
    )
    model = tf.keras.Sequential(
        [
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Full pipeline
    bb = get_backbone_submodel(model)
    assert bb is backbone

    conv_layer = find_last_conv_layer(bb)
    assert conv_layer.name == "e2e_conv2"

    img = np.random.rand(1, 96, 96, 3).astype(np.float32)
    heatmap = make_gradcam_heatmap(img, model, conv_layer)
    assert heatmap.ndim == 2
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

    # Overlay on a fake BGR image
    fake_bgr = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    overlay = overlay_heatmap(fake_bgr, heatmap)
    assert overlay.shape == fake_bgr.shape
    assert overlay.dtype == np.uint8
