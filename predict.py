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
from preprocessing import preprocess_image_array, preprocess_image_bytes
from utils.model_loader import load_cached_model

logger = logging.getLogger(__name__)


def preprocess_image(image_input: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Return an RGB normalized batch for a path, raw bytes, or BGR array."""
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
        image_input = image_path.read_bytes()

    try:
        if isinstance(image_input, bytes):
            return preprocess_image_bytes(image_input)

        if isinstance(image_input, np.ndarray):
            return preprocess_image_array(image_input)

        raise TypeError("image_input must be a file path, raw bytes, or numpy array.")
    except TypeError:
        raise
    except ValueError as exc:
        raise PreprocessingError(f"Failed to preprocess image: {exc}") from exc
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
) -> dict:
    """Run deepfake detection and return a normalized result dictionary."""
    source_path = str(image_input) if isinstance(image_input, (str, Path)) else None
    processed = preprocess_image(image_input)

    try:
        model = load_cached_model(model_path)
        prediction = model.predict(processed, verbose=0)
        label, confidence, raw_scores = decode_prediction(prediction)
    except ModelExecutionError:
        raise
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc, exc_info=True)
        raise ModelExecutionError(f"Model prediction failed: {exc}") from exc

    result = {
        "label": label,
        "confidence": confidence,
        "raw": raw_scores,
        "processed_image": processed,
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
    parser.add_argument("--json", dest="output_json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = []
    exit_code = 0

    for image_path in args.images:
        try:
            results.append(predict_image(image_path, model_path=args.model))
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
            {key: value for key, value in result.items() if key != "processed_image"}
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
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
