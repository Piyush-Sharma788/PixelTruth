import os
import sys
import logging
import numpy as np

from preprocessing import preprocess_image_bytes
from exceptions import (
    PreprocessingError,
    ModelExecutionError,
)

from utils.model_loader import load_cached_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_image(image_path):

    try:

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with open(image_path, "rb") as file_handle:

            image_bytes = file_handle.read()

        return preprocess_image_bytes(image_bytes)

    except Exception as e:

        logger.error(
            f"Image preprocessing failed: {e}",
            exc_info=True
        )

        raise PreprocessingError(str(e)) from e

# ---------------- PREDICTION ----------------

def predict_image(image_path):

    try:

        model = load_cached_model()

        image = preprocess_image(image_path)

        prediction = model.predict(image, verbose=0)

        class_label = np.argmax(prediction, axis=1)[0]

        confidence = float(np.max(prediction)) * 100

        label = "Fake" if class_label == 1 else "Real"

        print(f"Prediction : {label}")

        print(f"Confidence : {confidence:.1f}%")

        return label

    except PreprocessingError:

        print("Error preprocessing image.")

        sys.exit(1)

    except Exception as e:

        logger.error(
            f"Prediction failed: {e}",
            exc_info=True
        )

        raise ModelExecutionError(str(e)) from e

# ---------------- MAIN ----------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        raise SystemExit(
            "Usage: python predict.py <image_path>"
        )

    predict_image(sys.argv[1])