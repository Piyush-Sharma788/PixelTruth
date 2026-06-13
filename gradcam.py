import numpy as np
import cv2


def get_backbone_submodel(model):
    """Return the backbone sub-model used for Grad-CAM gradient computation.

    Why dynamic lookup?  The top-level model may be wrapped in a
    Sequential or Functional container whose first layer is *not*
    always the backbone (e.g. a Rescaling or InputLayer could be
    prepended).  Hardcoding ``model.layers[0]`` silently breaks
    Grad-CAM when the architecture changes.  This helper walks
    ``model.layers`` and picks the best candidate dynamically.

    Selection strategy
    ------------------
    1. Prefer the first nested ``tf.keras.Model`` whose own layers
       include at least one convolutional layer (the real backbone).
    2. If no conv-bearing sub-model exists, return any nested Model
       (covers unusual wrappers).
    3. If the top-level model itself contains conv layers directly
       (flat architecture with no nesting), return ``model`` as-is.
    4. Otherwise raise ``ValueError`` with an actionable message.
    """
    # Import TensorFlow lazily to avoid import-time side effects during tests
    import tensorflow as tf

    def _has_conv(m):
        """Return True if any layer in *m* has 'Conv' in its class name."""
        for layer in getattr(m, "layers", []):
            if "Conv" in layer.__class__.__name__:
                return True
        return False

    # --- Pass 1: nested sub-model with convolutional layers (best match) ---
    first_submodel = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if first_submodel is None:
                first_submodel = layer
            if _has_conv(layer):
                return layer

    # --- Pass 2: nested sub-model without conv layers (unusual wrapper) ---
    if first_submodel is not None:
        return first_submodel

    # --- Pass 3: flat model that itself has conv layers ---
    if _has_conv(model):
        return model

    # --- Nothing found — give an actionable error ---
    layer_types = [type(l).__name__ for l in model.layers]
    raise ValueError(
        "No backbone sub-model found in model.layers and the model "
        "contains no convolutional layers.  Grad-CAM requires at least "
        "one Conv layer for gradient computation.\n"
        f"  Layer types present: {layer_types}\n"
        "Ensure the model contains a nested tf.keras.Model backbone "
        "(e.g. Xception, EfficientNet) or direct Conv2D layers."
    )



def _forward_pass(container, input_tensor, stop_layer):
    import tensorflow as tf

    current = input_tensor
    conv_output = None
    for layer in getattr(container, "layers", []):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        sub_layers = getattr(layer, "layers", None)
        if sub_layers:
            current, conv_out = _forward_pass(layer, current, stop_layer)
            if conv_out is not None:
                conv_output = conv_out
        else:
            current = layer(current)
        if layer == stop_layer:
            conv_output = current
    return current, conv_output


def make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index=None):
    import tensorflow as tf

    if isinstance(last_conv_layer, str):
        def find_layer_by_name(m, name):
            try:
                return m.get_layer(name)
            except ValueError:
                pass
            for layer in getattr(m, "layers", []):
                if hasattr(layer, "layers"):
                    res = find_layer_by_name(layer, name)
                    if res is not None:
                        return res
            return None

        layer_obj = find_layer_by_name(model, last_conv_layer)
        if layer_obj is None:
            raise ValueError(f"No layer named '{last_conv_layer}' found in the model.")
        last_conv_layer = layer_obj

    try:
        _ = model(img_array)
    except Exception:
        pass

    with tf.GradientTape() as tape:
        predictions, conv_outputs = _forward_pass(model, img_array, last_conv_layer)
        if conv_outputs is None:
            raise ValueError(
                f"Layer {last_conv_layer.name} was not found during forward pass."
            )
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError(
            "tape.gradient() returned None for conv_outputs. "
            "The tensor was not watched or the computation graph is "
            "disconnected. Ensure the last_conv_layer is part of the "
            "model's computation graph."
        )
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 1e-10:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img