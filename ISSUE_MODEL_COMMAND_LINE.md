# [BUG] CLI argument `--model` is completely ignored by the prediction pipeline

## Summary
The PixelTruth CLI (`predict.py`) exposes a `--model` argument allowing users to evaluate deepfake detection models located at custom file paths. However, the prediction pipeline completely ignores this argument, always loading the default cached model (either the hardcoded `deepfake_detection_model.h5` or the path specified in the `PIXELTRUTH_MODEL_PATH` environment variable).

This prevents researchers and developers from using the CLI to run inference on custom/fine-tuned weights without globally overriding environment variables.

---

## Evidence

### 1. CLI parses and passes `--model`
In `predict.py::main`, the `--model` argument is parsed and passed to `predict_image` as the `model_path` keyword argument:
```python
result = predict_image(
    image_path,
    model_path=args.model
)
```

### 2. `predict_image` ignores `model_path`
Inside `predict.py::predict_image`, the keyword argument `model_path` is never utilized or forwarded to the model loader. Instead, it always calls `load_cached_model()` without passing the path:
```python
def predict_image(
    image_input: str | Path | bytes | np.ndarray,
    model_path: str | None = None
) -> dict:
    image = preprocess_image(image_input)

    try:
        # Cached lazy-loaded model
        model = load_cached_model(get_model_mtime())
        prediction = model.predict(image, verbose=0)
```

### 3. Caching loader is hardcoded to global configuration
Inside `utils/model_loader.py`, `load_cached_model` is decorated with `@st.cache_resource` and is hardcoded to load from the static module-level `MODEL_PATH`:
```python
MODEL_PATH = get_model_path() # Resolved at import time

@st.cache_resource
def load_cached_model(model_mtime):
    model_file_path = ensure_model_file(
        model_path=MODEL_PATH,
        ...
    )
    model = load_model(model_file_path)
```

---

## Steps To Reproduce
1. Train a model and save it to a custom location, e.g., `fine_tuned_weights.h5`.
2. Run predictions using the CLI and specify the custom weights path:
   ```bash
   python predict.py --model fine_tuned_weights.h5 sample_image.jpg
   ```
3. Observe that the logs/system will still load `deepfake_detection_model.h5` (or whatever the environment variable dictates) instead of using the specified `fine_tuned_weights.h5`.

---

## Expected Result
When `--model <PATH>` is supplied via the CLI:
1. The prediction pipeline should resolve the custom path.
2. If the model at the custom path is already loaded, it can retrieve it from cache; otherwise, it should load the custom model.
3. The custom model should be used to run prediction on the image.

---

## Suggested Resolution
Modify `predict_image` and `load_cached_model` to accept and correctly process `model_path` overrides:
1. In `utils/model_loader.py`, modify `load_cached_model` to take an optional `model_path` parameter:
   ```python
   @st.cache_resource
   def load_cached_model(model_mtime, model_path=None):
       target_path = model_path or MODEL_PATH
       model_file_path = ensure_model_file(
           model_path=target_path,
           ...
       )
       return load_model(model_file_path)
   ```
2. In `predict.py::predict_image`, pass `model_path` directly to `load_cached_model`:
   ```python
   model = load_cached_model(get_model_mtime(), model_path=model_path)
   ```
