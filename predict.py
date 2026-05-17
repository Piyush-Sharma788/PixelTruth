import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256

MODEL_PATH = get_model_path()
MODEL_URL = get_model_url()
MODEL_SHA256 = get_model_sha256()
_model = None


def load_deepfake_model():
    global _model
    if _model is not None:
        return _model

    model_file_path = ensure_model_file(
        model_path=MODEL_PATH,
        model_url=MODEL_URL,
        model_sha256=MODEL_SHA256,
        download_if_missing=True,
    )
    _model = load_model(model_file_path)
    return _model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image nahi mili: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # fix: cv2.imread returns BGR; model expects RGB (trained via PIL/ImageDataGenerator)
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict_image(image_path):
    model = load_deepfake_model()
    image = preprocess_image(image_path)
    prediction = model.predict(image, verbose=0)
    print(f"Raw prediction: {prediction}")
    class_label = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction)) * 100
    # Class 1 = Fake, Class 0 = Real (dataset mapping)
    label = "Fake" if class_label == 1 else "Real"
    print(f"Prediction  : {label}")
    print(f"Confidence  : {confidence:.1f}%")
    return label

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict.py <image_path>")

    predict_image(sys.argv[1])
