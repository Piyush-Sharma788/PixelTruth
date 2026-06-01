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



def _find_layer_by_name_recursive(model, layer_name):
    """Recursively search for a layer by name in model and its nested layers."""
    try:
        return model.get_layer(layer_name)
    except ValueError:
        pass
    
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "layers"):
            result = _find_layer_by_name_recursive(layer, layer_name)
            if result is not None:
                return result
    return None


def _find_last_conv_layer_recursive(model):
    """Recursively search for the last convolutional layer in the model."""
    layers = getattr(model, "layers", None) or []
    for layer in reversed(layers):
        # Recursively search in nested models first
        if hasattr(layer, "layers") and getattr(layer, "layers"):
            try:
                return _find_last_conv_layer_recursive(layer)
            except ValueError:
                continue
        
        # Check if this layer is a convolutional layer
        if "Conv" in layer.__class__.__name__:
            return layer
    
    # Try flattened layers if available
    try:
        for layer in reversed(list(model._flatten_layers())):
            if "Conv" in layer.__class__.__name__:
                return layer
    except Exception:
        pass
    
    raise ValueError("No convolutional layer found in the model")


def make_gradcam_heatmap(img_array, model, last_conv_layer=None, pred_index=None):
    """Generate Grad-CAM heatmap for a given image and model.
    
    Args:
        img_array: Input image array (batch of 1), shape (1, H, W, C)
        model: Full trained Keras model for classification
        last_conv_layer: Conv layer object or layer name (string).
                        If None, finds the last conv layer automatically.
        pred_index: Target class index for gradient computation.
                   If None, defaults to argmax(predictions).
    
    Returns:
        heatmap: Normalized heatmap array, shape (H', W')
                 where H', W' are conv output spatial dimensions.
    
    Raises:
        ValueError: If no convolutional layer found or layer not in model.
    """
    # Import TensorFlow lazily to avoid import-time side effects during tests
    import tensorflow as tf

    # Resolve layer reference: string → layer object
    if isinstance(last_conv_layer, str):
        layer_obj = _find_layer_by_name_recursive(model, last_conv_layer)
        if layer_obj is None:
            raise ValueError(
                f"No layer named '{last_conv_layer}' found in the model. "
                "Provide a valid layer name or layer object."
            )
        last_conv_layer = layer_obj
    elif last_conv_layer is None:
        # Auto-find the last conv layer
        try:
            last_conv_layer = _find_last_conv_layer_recursive(model)
        except ValueError:
            raise ValueError(
                "No convolutional layer found in the model. "
                "Grad-CAM requires at least one Conv layer for gradient computation. "
                "Provide an explicit last_conv_layer or ensure the model contains Conv layers."
            )

    # Ensure the model is built on the input structure
    try:
        _ = model(img_array)
    except Exception:
        pass

    # Get output tensors, handling both symbolic and non-symbolic outputs
    try:
        conv_output = last_conv_layer.output
    except Exception:
        try:
            conv_output = last_conv_layer.outputs[0]
        except Exception:
            raise ValueError(
                f"Unable to retrieve output tensor from layer '{last_conv_layer.name}'. "
                "Ensure it is a properly configured Keras layer."
            )

    try:
        model_output = model.output
    except Exception:
        try:
            model_output = model.outputs[0]
        except Exception:
            raise ValueError(
                "Unable to retrieve output tensor from the model. "
                "Ensure the model has a well-defined output."
            )

    # Build the Grad-CAM model: returns both conv output and final prediction
    try:
        grad_model = tf.keras.models.Model(
            model.inputs,
            [conv_output, model_output],
        )
    except Exception as e:
        raise ValueError(
            f"Failed to build Grad-CAM model with layer '{last_conv_layer.name}'. "
            f"Error: {str(e)}"
        )

    # Forward pass and gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Determine target class index
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        # Extract gradients of target class w.r.t. conv outputs
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        raise ValueError(
            f"Gradient computation failed for layer '{last_conv_layer.name}'. "
            "Ensure the layer is connected to the model output and supports gradients."
        )

    # Apply spatial pooling of gradients and generate heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    
    # Normalize heatmap to [0, 1]
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