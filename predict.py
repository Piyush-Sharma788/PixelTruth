"""
PixelTruth — unified inference pipeline.

This module is the **single source of truth** for image preprocessing and
deepfake prediction.  Both the Streamlit dashboard (``app.py``) and the CLI
import from here, ensuring identical behaviour regardless of the entry-point.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from preprocessing import preprocess_image_bytes, preprocess_image_array
import logging

from exceptions import (
    PreprocessingError,
    ModelExecutionError,
)

from utils.model_loader import load_cached_model, get_model_mtime
from inference import decode_prediction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified preprocessing — accepts file paths, numpy arrays, or raw bytes
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

def preprocess_image(image_input) -> np.ndarray:
    """Preprocess an image for model inference.

    Accepts multiple input types so that every caller (CLI, Streamlit, tests)
    can use a single function:

    * **str / Path** — filesystem path; the file is read and decoded.
    * **bytes** — raw image bytes (e.g. from ``UploadedFile.read()``).
    * **np.ndarray** — a BGR image already loaded into memory.

    Parameters
    ----------
    image_input:
        The image to preprocess. See above for accepted types.

    Returns
    -------
    np.ndarray
        Shape ``(1, H, W, 3)`` with values in ``[0, 255]``, channels in RGB
        order — ready to be passed directly to ``model.predict()``.

    Raises
    ------
    FileNotFoundError
        When a path string is provided but the file does not exist.
    ValueError
        When a path has an unsupported extension, or bytes cannot be decoded.
    PreprocessingError
        When preprocessing fails for any other reason.
    TypeError
        When *image_input* is not a supported type.
    """

    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = image_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        try:
            with open(image_path, "rb") as file_handle:
                image_bytes = file_handle.read()
            return preprocess_image_bytes(image_bytes)
        except Exception as e:
            logger.error(
                f"Image preprocessing failed for {image_path}: {e}",
                exc_info=True
            )
            raise PreprocessingError(
                f"Failed to preprocess image: {str(e)}"
            ) from e

    elif isinstance(image_input, bytes):
        try:
            return preprocess_image_bytes(image_input)
        except Exception as e:
            logger.error(
                f"Image preprocessing failed for bytes: {e}",
                exc_info=True
            )
            raise PreprocessingError(
                f"Failed to preprocess image bytes: {str(e)}"
            ) from e

    elif isinstance(image_input, np.ndarray):
        try:
            return preprocess_image_array(image_input)
        except Exception as e:
            logger.error(
                f"Image preprocessing failed for numpy array: {e}",
                exc_info=True
            )
            raise PreprocessingError(
                f"Failed to preprocess image array: {str(e)}"
            ) from e

    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


# ---------------------------------------------------------------------------
# Unified prediction
# ---------------------------------------------------------------------------

def predict_image(
    image_input: str | Path | bytes | np.ndarray,
    model_path: str | None = None
) -> dict:
    """Run deepfake detection on a single image.

    Parameters
    ----------
    image_input:
        Path, raw bytes, or BGR numpy array to classify.

    model_path:
        Optional override for the model file location.

    Returns
    -------
    dict
        ``{"label": "Real"|"Fake", "confidence": float, "raw": list[float],
          "processed_image": np.ndarray}``

        * ``confidence`` is a **float in [0, 1]** (NOT a percentage).
        * ``processed_image`` is the preprocessed tensor used for inference.
        * For CLI callers the dict also includes ``"image": str`` when a path
          was provided.

    Raises
    ------
    FileNotFoundError
        When path input does not exist on disk.

    ValueError
        When file extension is not supported or bytes cannot be decoded.

    PreprocessingError
        When the image cannot be decoded or preprocessed.

    ModelExecutionError
        When model inference fails.
    """

    image = preprocess_image(image_input)

    try:
        # Cached lazy-loaded model
        model = load_cached_model(get_model_mtime())
        prediction = model.predict(image, verbose=0)

    except (
        PreprocessingError,
        FileNotFoundError,
        ValueError
    ):
        raise

    except Exception as e:
        logger.error(
            f"Model prediction failed: {e}",
            exc_info=True
        )
        raise ModelExecutionError(
            f"Model prediction failed: {str(e)}"
        ) from e

    label, confidence = decode_prediction(prediction)

    result: dict = {
        "label": label,
        "confidence": confidence,
        "raw": prediction[0].tolist(),
        "processed_image": image,
    }

    # Include path metadata when the input was a file path
    if isinstance(image_input, (str, Path)):
        result["image"] = str(image_input)

    return result


# ---------------------------------------------------------------------------
# Convenience wrappers (backward-compat for app.py)
# ---------------------------------------------------------------------------


def predict_image_tuple(image_input):
    """Thin wrapper returning ``(label, confidence, processed_image)``.

    Used by ``app.py`` which was originally built around a tuple return value.
    If no model is loaded, returns ``(None, None, None)``.
    """
    try:
        result = predict_image(image_input)
    except Exception:
        # When model loading fails altogether, mirror the old None-tuple.
        return None, None, None
    return result["label"], result["confidence"], result["processed_image"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "PixelTruth — deepfake image detector.\n"
            "Classifies one or more images as Real or Fake."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py photo.jpg\n"
            "  python predict.py img1.jpg img2.png --json\n"
            "  python predict.py --model /weights/model.h5 photo.jpg\n\n"
            "Environment variables:\n"
            "  PIXELTRUTH_MODEL_PATH   path to model file\n"
            "  PIXELTRUTH_MODEL_URL    URL to download model if missing\n"
            "  PIXELTRUTH_MODEL_SHA256 expected SHA-256 of the model file"
        ),
    )

    parser.add_argument(
        "images",
        metavar="IMAGE",
        nargs="+",
        help="path(s) to image file(s) to classify",
    )

    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help=(
            "path to the .h5 model file "
            "(default: $PIXELTRUTH_MODEL_PATH or "
            "'deepfake_detection_model.h5')"
        ),
    )

    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="print results as JSON (useful for scripting)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress informational messages; only print results",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 if any image fails."""

    parser = build_parser()

    args = parser.parse_args(argv)

    results = []

    exit_code = 0

    for image_path in args.images:

        try:

            result = predict_image(
                image_path,
                model_path=args.model
            )

            results.append(result)

        except (
            FileNotFoundError,
            ValueError,
            PreprocessingError,
            ModelExecutionError
        ) as exc:

            error_result = {
                "image": image_path,
                "error": str(exc),
            }

            results.append(error_result)

            exit_code = 1

            if not args.quiet:
                print(f"[ERROR] {exc}", file=sys.stderr)

    if args.output_json:

        print(
            json.dumps(
                results if len(results) > 1 else results[0],
                indent=2
            )
        )

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