# CHANGED: removed numpy, cv2, tensorflow.keras imports — no longer needed with HF pipeline
from PIL import Image
from transformers import pipeline

# CHANGED: replaced load_model('deepfake_detection_model.h5') with HF pipeline that auto-downloads on first run
_pipe = pipeline("image-classification", model="prithivMLmods/deepfake-detector-model-v1")

# CHANGED: signature changed from image_path (str) to image (PIL.Image) to match app.py's upload flow
def predict_image(image: Image.Image) -> tuple[str, float]:
    # CHANGED: HF pipeline handles resizing/preprocessing internally; no manual cv2 resize needed
    results = _pipe(image)
    top = results[0]
    raw_label: str = top["label"]
    confidence: float = float(top["score"])

    # CHANGED: normalise label to "Real"/"Fake" regardless of the model's internal label strings
    label = "Real" if "real" in raw_label.lower() else "Fake"
    return label, confidence
