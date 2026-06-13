"""Unified inference pipeline using Hugging Face Transformers (SigLIP)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import SUPPORTED_EXTENSIONS, HF_MODEL_NAME
from exceptions import ModelExecutionError, PreprocessingError
from preprocessing import decode_image_bytes, detect_and_crop_face
from utils.model_loader import load_cached_model

logger = logging.getLogger(__name__)


def preprocess_image(image_input: str | Path | bytes | np.ndarray) -> Image.Image:
    """Return a PIL RGB image (face-cropped if detected) for the HF processor."""
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
        if image_input.ndim == 2:
            bgr_image = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            bgr_image = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        elif image_input.shape[2] == 1:
            bgr_image = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        else:
            bgr_image = image_input
    else:
        raise TypeError("image_input must be a file path, raw bytes, or numpy array.")

    try:
        face_image, _ = detect_and_crop_face(bgr_image)
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    except Exception as exc:
        logger.error("Image preprocessing failed: %s", exc, exc_info=True)
        raise PreprocessingError(f"Failed to preprocess image: {exc}") from exc


def decode_prediction(logits: np.ndarray) -> tuple[str, float, list[float]]:
    """Convert HF model logits to label, confidence, and raw scores."""
    import torch

    scores = np.asarray(logits, dtype=float).reshape(-1)
    probs = torch.softmax(torch.from_numpy(scores), dim=0).numpy()

    if probs.size != 2:
        raise ModelExecutionError(
            f"Unsupported model output shape: {np.asarray(logits).shape}. "
            f"Expected 2 classes (fake, real)."
        )

    class_index = int(np.argmax(probs))
    label = "Real" if class_index == 1 else "Fake"
    return label, float(probs[class_index]), probs.tolist()


def predict_image(
    image_input: str | Path | bytes | np.ndarray,
    temperature: float = 1.0,
) -> dict:
    """Run deepfake detection via Hugging Face model and return a normalized result dictionary."""
    source_path = str(image_input) if isinstance(image_input, (str, Path)) else None

    pil_image = preprocess_image(image_input)

    try:
        processor, model = load_cached_model()

        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.detach().numpy()

        raw_confidence = float(np.max(logits))

        scaled_logits = logits / temperature
        label, confidence, raw_scores = decode_prediction(scaled_logits)

        face_bgr = np.array(pil_image)[:, :, ::-1]
    except ModelExecutionError:
        raise
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc, exc_info=True)
        raise ModelExecutionError(f"Model prediction failed: {exc}") from exc

    result = {
        "label": label,
        "confidence": confidence,
        "raw_confidence": raw_confidence,
        "raw_prediction": logits,
        "raw": raw_scores,
        "processed_image": np.array(pil_image),
        "face_detected": True,
        "face_box": None,
        "face_image": face_bgr,
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
        description="PixelTruth deepfake image detector (Hugging Face).",
    )
    parser.add_argument("images", metavar="IMAGE", nargs="+", help="image path(s)")
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
            res = predict_image(image_path, temperature=args.temperature)
            results.append(res)

            if not args.quiet and not args.output_json:
                if res["confidence"] < args.threshold:
                    print(
                        f"[WARNING] Low-confidence prediction for {image_path}: "
                        f"{res['confidence'] * 100:.1f}% (threshold: {args.threshold * 100:.1f}%)",
                        file=sys.stderr,
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
            {key: value for key, value in result.items() if key not in ("processed_image", "face_image")}
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
