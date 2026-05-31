import numpy as np
import cv2


def get_backbone_submodel(model):
    """Return the first nested tf.keras.Model found in model.layers."""
    # Import TensorFlow lazily to avoid import-time side effects during tests
    import tensorflow as tf

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise ValueError(
        "No backbone sub-model found in model.layers. "
        "Ensure the model contains a nested tf.keras.Model."
    )



def make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index=None):
    # Import TensorFlow lazily to avoid import-time side effects during tests
    import tensorflow as tf

    # If last_conv_layer is passed as a string, retrieve the layer object recursively
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

    # Ensure the model has been called/built on the input structure so that input/output nodes are initialized
    try:
        _ = model(img_array)
    except Exception:
        pass

    # 1. Identify if the last_conv_layer is nested inside a sub-model
    sub_model = None
    for layer in getattr(model, "layers", []):
        if layer == last_conv_layer:
            break
        elif hasattr(layer, "layers") and any(l == last_conv_layer for l in getattr(layer, "layers", [])):
            sub_model = layer
            break

    # 2. Reconstruct functional/sequential outputs accordingly
    if sub_model is not None:
        if not isinstance(sub_model, tf.keras.Sequential):
            try:
                sub_conv_output = last_conv_layer.output
            except Exception:
                try:
                    sub_conv_output = last_conv_layer.outputs[0]
                except Exception:
                    sub_conv_output = last_conv_layer.get_output_at(0)

            try:
                sub_model_output = sub_model.output
            except Exception:
                try:
                    sub_model_output = sub_model.outputs[0]
                except Exception:
                    sub_model_output = sub_model.get_output_at(0)

            grad_sub_model = tf.keras.models.Model(
                sub_model.inputs,
                [sub_conv_output, sub_model_output]
            )
        else:
            grad_sub_model = None

        with tf.GradientTape() as tape:
            current = img_array
            conv_outputs = None
            for layer in model.layers:
                if layer == sub_model:
                    if grad_sub_model is not None:
                        conv_outputs, current = grad_sub_model(current)
                    else:
                        # Sequential sub-model execution
                        for sub_layer in sub_model.layers:
                            current = sub_layer(current)
                            if sub_layer == last_conv_layer:
                                conv_outputs = current
                elif layer == last_conv_layer:
                    current = layer(current)
                    conv_outputs = current
                else:
                    current = layer(current)
            predictions = current

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    else:
        # Fallback to standard flat model construction
        try:
            conv_output = last_conv_layer.output
        except Exception:
            try:
                conv_output = last_conv_layer.outputs[0]
            except Exception:
                conv_output = last_conv_layer.get_output_at(0)

        try:
            model_output = model.output
        except Exception:
            try:
                model_output = model.outputs[0]
            except Exception:
                model_output = model.get_output_at(0)

        grad_model = tf.keras.models.Model(
            model.inputs,
            [
                conv_output,
                model_output,
            ],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
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