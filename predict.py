"""Unified inference pipeline used by the CLI and FastAPI application."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from config import SUPPORTED_EXTENSIONS
from exceptions import ModelExecutionError, PreprocessingError
from preprocessing import preprocess_image_array, preprocess_image_bytes, decode_image_bytes, detect_and_crop_face
from utils.model_loader import load_cached_model, get_model_mtime

logger = logging.getLogger(__name__)


def preprocess_image(image_input: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Return an RGB normalized batch for a path, raw bytes, or BGR array after face detection."""
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file extension '{image_path.suffix.lower()}'. "
                f"Supported: {supported}"
            )
        raw_bytes = image_path.read_bytes()
        bgr_image = decode_image_bytes(raw_bytes)
    elif isinstance(image_input, bytes):
        bgr_image = decode_image_bytes(image_input)
    elif isinstance(image_input, np.ndarray):
        bgr_image = image_input
    else:
        raise TypeError("image_input must be a file path, raw bytes, or numpy array.")

    try:
        face_image, _ = detect_and_crop_face(bgr_image)
        return preprocess_image_array(face_image)
    except Exception as exc:
        logger.error("Image preprocessing failed: %s", exc, exc_info=True)
        raise PreprocessingError(f"Failed to preprocess image: {exc}") from exc


def decode_prediction(prediction: np.ndarray) -> tuple[str, float, list[float]]:
    """Convert sigmoid or two-class softmax output to a label and confidence."""
    scores = np.asarray(prediction, dtype=float).reshape(-1)
    if scores.size == 1:
        real_probability = float(scores[0])
        if not 0.0 <= real_probability <= 1.0:
            raise ModelExecutionError("Model returned a probability outside [0, 1].")
        # Training directories are alphabetic: class 0 = fake, class 1 = real.
        if real_probability >= 0.5:
            return "Real", real_probability, scores.tolist()
        return "Fake", 1.0 - real_probability, scores.tolist()

    if scores.size == 2:
        class_index = int(np.argmax(scores))
        label = "Real" if class_index == 1 else "Fake"
        return label, float(scores[class_index]), scores.tolist()

    raise ModelExecutionError(
        f"Unsupported model output shape: {np.asarray(prediction).shape}."
    )


def predict_image(
    image_input: str | Path | bytes | np.ndarray,
    model_path: str | None = None,
    temperature: float = 1.0,
) -> dict:
    """Run deepfake detection and return a normalized result dictionary."""
    source_path = str(image_input) if isinstance(image_input, (str, Path)) else None

    # Get BGR image
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file extension '{image_path.suffix.lower()}'. "
                f"Supported: {supported}"
            )
        raw_bytes = image_path.read_bytes()
        bgr_image = decode_image_bytes(raw_bytes)
    elif isinstance(image_input, bytes):
        bgr_image = decode_image_bytes(image_input)
    elif isinstance(image_input, np.ndarray):
        bgr_image = image_input
    else:
        raise TypeError("image_input must be a file path, raw bytes, or numpy array.")

    try:
        face_image, face_box = detect_and_crop_face(bgr_image)
        processed = preprocess_image_array(face_image)
    except Exception as exc:
        logger.warning("Face detection failed, falling back to full image: %s", exc)
        face_image = bgr_image
        face_box = None
        try:
            processed = preprocess_image_array(face_image)
        except Exception as inner_exc:
            logger.error("Image preprocessing failed: %s", inner_exc, exc_info=True)
            raise PreprocessingError(f"Failed to preprocess image: {inner_exc}") from inner_exc

    try:
        model = load_cached_model(get_model_mtime(model_path), model_path=model_path)
        prediction = model.predict(processed, verbose=0)
        
        # Decode raw prediction for raw confidence
        _, raw_confidence, _ = decode_prediction(prediction)
        
        # Apply temperature scaling for calibrated confidence
        from calibration import temperature_scale
        calibrated_prediction = temperature_scale(prediction, temperature=temperature)
        label, confidence, raw_scores = decode_prediction(calibrated_prediction)
    except ModelExecutionError:
        raise
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc, exc_info=True)
        raise ModelExecutionError(f"Model prediction failed: {exc}") from exc

    result = {
        "label": label,
        "confidence": confidence,
        "raw_confidence": raw_confidence,
        "raw_prediction": prediction,
        "raw": raw_scores,
        "processed_image": processed,
        "face_detected": face_box is not None,
        "face_box": face_box,
        "face_image": face_image,
    }
    if source_path is not None:
        result["image"] = source_path
    return result


def predict_image_tuple(image_input):
    """Backward compatible wrapper returning label, confidence, and image batch."""
    try:
        result = predict_image(image_input)
    except Exception:
        return None, None, None
    return result["label"], result["confidence"], result["processed_image"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="PixelTruth deepfake image detector.",
    )
    parser.add_argument("images", metavar="IMAGE", nargs="+", help="image path(s)")
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="path to the Keras model file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="confidence calibration temperature (>0, default: 1.5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="confidence threshold for low-confidence warnings (default: 0.70)",
    )
    parser.add_argument("--json", dest="output_json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = []
    exit_code = 0

    for image_path in args.images:
        try:
            res = predict_image(
                image_path,
                model_path=args.model,
                temperature=args.temperature,
            )
            results.append(res)
            
            # Print low confidence warning if applicable
            if not args.quiet and not args.output_json:
                if res["confidence"] < args.threshold:
                    print(
                        f"[WARNING] Low-confidence prediction for {image_path}: "
                        f"{res['confidence'] * 100:.1f}% (threshold: {args.threshold * 100:.1f}%)",
                        file=sys.stderr
                    )
        except (
            FileNotFoundError,
            TypeError,
            ValueError,
            PreprocessingError,
            ModelExecutionError,
        ) as exc:
            results.append({"image": image_path, "error": str(exc)})
            exit_code = 1
            if not args.quiet:
                print(f"[ERROR] {exc}", file=sys.stderr)

    if args.output_json:
        serializable_results = [
            {key: value for key, value in result.items() if key not in ("processed_image", "face_image", "raw_prediction")}
            for result in results
        ]
        output = (
            serializable_results
            if len(serializable_results) > 1
            else serializable_results[0]
        )
        print(json.dumps(output, indent=2))
    else:
        for result in results:
            if "error" in result:
                continue
            if not args.quiet:
                print(f"\nImage      : {result['image']}")
                print(f"Raw output : {result['raw']}")
            print(f"Prediction : {result['label']}")
            print(f"Confidence : {result['confidence'] * 100:.1f}%")
            if "raw_confidence" in result and abs(result["raw_confidence"] - result["confidence"]) > 1e-4:
                print(f"Raw Conf.  : {result['raw_confidence'] * 100:.1f}%")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
