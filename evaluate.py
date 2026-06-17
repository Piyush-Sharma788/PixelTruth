"""
Offline model evaluation harness for PixelTruth.

Loads the trained deepfake-detection model, runs it against a labelled test
dataset, computes real performance metrics using scikit-learn, and serializes
the results to a JSON cache file that ``metrics.py`` can render in Streamlit.

Usage
-----
CLI::

    python evaluate.py --test-dir path/to/test
    python evaluate.py --test-dir path/to/test --output results.json
    python evaluate.py --test-dir path/to/test --model custom.h5 --batch-size 64

Programmatic::

    from evaluate import run_evaluation
    from tensorflow.keras.models import load_model

    model = load_model("deepfake_detection_model.h5")
    results = run_evaluation(model, "path/to/test")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from preprocessing import preprocess_image_from_path
from config import SUPPORTED_EXTENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` while guarding zero division."""
    return float(numerator / denominator) if denominator else 0.0


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return classification accuracy for aligned 1D label arrays."""
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a 2x2 confusion matrix for binary labels ``0`` and ``1``."""
    matrix = np.zeros((2, 2), dtype=int)
    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[true_label, pred_label] += 1
    return matrix


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the weighted-average metrics used by the evaluation output."""
    total = int(y_true.size)
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for label in (0, 1):
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1_score = _safe_divide(2 * precision * recall, precision + recall)

        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1_score * support

    if total == 0:
        return {"weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}}

    return {
        "weighted avg": {
            "precision": weighted_precision / total,
            "recall": weighted_recall / total,
            "f1-score": weighted_f1 / total,
        }
    }


def _roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute FPR/TPR arrays for binary labels using descending score order."""
    if y_true.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(y_scores, kind="mergesort")[::-1]
    sorted_true = y_true[order].astype(int)
    sorted_scores = y_scores[order]

    distinct_score_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_score_indices, sorted_true.size - 1]

    true_positives = np.cumsum(sorted_true == 1)[threshold_indices]
    false_positives = (threshold_indices + 1) - true_positives

    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]

    positive_count = int(np.sum(sorted_true == 1))
    negative_count = int(np.sum(sorted_true == 0))

    if positive_count == 0:
        tpr = np.zeros_like(true_positives, dtype=float)
    else:
        tpr = true_positives / positive_count

    if negative_count == 0:
        fpr = np.zeros_like(false_positives, dtype=float)
    else:
        fpr = false_positives / negative_count

    return fpr.astype(float), tpr.astype(float)


def _auc(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute the area under a curve using the trapezoidal rule."""
    return float(np.trapz(y_values, x_values))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _collect_image_paths(test_data_dir: str) -> tuple[list[str], list[int]]:
    """Walk ``test_data_dir/real/`` and ``test_data_dir/fake/`` to collect
    image file paths and ground-truth labels.

    Returns
    -------
    paths : list[str]
        Absolute paths to image files.
    labels : list[int]
        Ground-truth labels (0 = Fake, 1 = Real), matching the Keras
        alphabetical class ordering used during training.

    Raises
    ------
    FileNotFoundError
        When the test directory or required subdirectories don't exist.
    ValueError
        When a class subdirectory contains no supported images.
    """
    base = Path(test_data_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_data_dir}")

    real_dir = base / "real"
    fake_dir = base / "fake"

    if not real_dir.is_dir():
        raise FileNotFoundError(
            f"Missing 'real/' subdirectory in {test_data_dir}. "
            f"Expected structure: {test_data_dir}/real/ and {test_data_dir}/fake/"
        )
    if not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Missing 'fake/' subdirectory in {test_data_dir}. "
            f"Expected structure: {test_data_dir}/real/ and {test_data_dir}/fake/"
        )

    paths: list[str] = []
    labels: list[int] = []

    for class_dir, label in [(real_dir, 1), (fake_dir, 0)]:
        class_paths = sorted(
            str(p)
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(
                f"No supported images found in {class_dir}. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))

    return paths, labels


def run_evaluation(
    model,
    test_data_dir: str,
    output_path: str = "metrics_cache.json",
    batch_size: int = 32,
) -> dict:
    """Evaluate the model against a labelled test dataset and save results.

    Parameters
    ----------
    model
        A loaded Keras model with a ``.predict()`` method.
    test_data_dir : str
        Path to directory containing ``real/`` and ``fake/`` subdirectories.
    output_path : str
        Where to write the JSON results cache.
    batch_size : int
        Number of images to batch during prediction.

    Returns
    -------
    dict
        The evaluation results dictionary (same content as the written JSON).
    """
    # 1. Collect images and labels
    image_paths, y_true = _collect_image_paths(test_data_dir)
    total_images = len(image_paths)
    logger.info(f"Found {total_images} images in {test_data_dir}")

    # 2. Preprocess and predict in batches
    y_prob_all: list[np.ndarray] = []
    skipped = 0

    # We need to track which images were successfully processed
    valid_y_true: list[int] = []

    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_labels = y_true[i : i + batch_size]
        batch_tensors: list[np.ndarray] = []
        batch_valid_labels: list[int] = []

        for path, label in zip(batch_paths, batch_labels):
            try:
                tensor = preprocess_image_from_path(path)
                batch_tensors.append(tensor)
                batch_valid_labels.append(label)
            except Exception as exc:
                logger.warning(f"Skipping {path}: {exc}")
                skipped += 1

        if batch_tensors:
            batch_input = np.concatenate(batch_tensors, axis=0)
            predictions = model.predict(batch_input, verbose=0)
            y_prob_all.append(predictions)
            valid_y_true.extend(batch_valid_labels)

        processed = min(i + batch_size, total_images)
        logger.info(f"Progress: {processed}/{total_images} images processed")

    if skipped:
        logger.warning(f"Skipped {skipped}/{total_images} corrupt/unreadable images")

    if not y_prob_all:
        raise ValueError("No images could be processed. Check your test dataset.")

    # 3. Aggregate predictions
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_true_arr = np.array(valid_y_true)

    # Predicted class: argmax across softmax outputs
    y_pred = np.argmax(y_prob, axis=1)

    # Probability of the positive class (Fake = class 1) for ROC
    y_scores = y_prob[:, 1]

    # 4. Compute metrics with scikit-learn
    acc = _accuracy_score(y_true_arr, y_pred)

    report = _classification_report(y_true_arr, y_pred)

    cm = _confusion_matrix(y_true_arr, y_pred)

    fpr, tpr = _roc_curve(y_true_arr, y_scores)
    roc_auc = _auc(fpr, tpr)

    # Per-class statistics
    n_real = int(np.sum(y_true_arr == 0))
    n_fake = int(np.sum(y_true_arr == 1))
    real_correct = int(cm[0, 0])
    fake_correct = int(cm[1, 1])

    results = {
        "accuracy": round(acc * 100, 2),
        "precision": round(report["weighted avg"]["precision"] * 100, 2),
        "recall": round(report["weighted avg"]["recall"] * 100, 2),
        "f1": round(report["weighted avg"]["f1-score"] * 100, 2),
        "auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "class_statistics": {
            "Real": {
                "total_samples": n_real,
                "correctly_classified": real_correct,
                "misclassified": n_real - real_correct,
                "class_accuracy": round(real_correct / n_real * 100, 1) if n_real else 0.0,
            },
            "Fake": {
                "total_samples": n_fake,
                "correctly_classified": fake_correct,
                "misclassified": n_fake - fake_correct,
                "class_accuracy": round(fake_correct / n_fake * 100, 1) if n_fake else 0.0,
            },
        },
        "dataset_distribution": {
            "Real": n_real,
            "Fake": n_fake,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(getattr(model, "name", "unknown")),
        "test_data_dir": str(test_data_dir),
        "total_images": len(valid_y_true),
        "skipped_images": skipped,
    }

    # 5. Serialize to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "PixelTruth — offline model evaluation harness.\n"
            "Evaluates the deepfake detection model against a labelled test\n"
            "dataset and writes performance metrics to a JSON cache file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected test directory structure:\n"
            "  test_data/\n"
            "  ├── real/\n"
            "  │   ├── img001.jpg\n"
            "  │   └── ...\n"
            "  └── fake/\n"
            "      ├── img001.jpg\n"
            "      └── ...\n\n"
            "Examples:\n"
            "  python evaluate.py --test-dir path/to/test\n"
            "  python evaluate.py --test-dir path/to/test --output eval.json\n"
            "  python evaluate.py --test-dir path/to/test --model custom.h5\n"
        ),
    )

    parser.add_argument(
        "--test-dir",
        required=True,
        metavar="DIR",
        help="path to test dataset directory (must contain real/ and fake/ subdirs)",
    )

    parser.add_argument(
        "--output",
        default="metrics_cache.json",
        metavar="PATH",
        help="output path for the JSON metrics cache (default: metrics_cache.json)",
    )

    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help=(
            "path to the .h5 model file "
            "(default: $PIXELTRUTH_MODEL_PATH or 'deepfake_detection_model.h5')"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="prediction batch size (default: 32)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        # Import model loading lazily to avoid TF side-effects during --help
        from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256
        from tensorflow.keras.models import load_model

        model_path = args.model or get_model_path()
        model_file = ensure_model_file(
            model_path=model_path,
            model_url=get_model_url(),
            model_sha256=get_model_sha256(),
            download_if_missing=True,
        )

        print(f"Loading model from {model_file}...")
        model = load_model(model_file)

        print(f"Evaluating against {args.test_dir}...")
        results = run_evaluation(
            model=model,
            test_data_dir=args.test_dir,
            output_path=args.output,
            batch_size=args.batch_size,
        )

        print(f"\n{'='*50}")
        print(f"  Evaluation Complete")
        print(f"{'='*50}")
        print(f"  Images evaluated : {results['total_images']}")
        print(f"  Images skipped   : {results['skipped_images']}")
        print(f"  Accuracy         : {results['accuracy']:.1f}%")
        print(f"  Precision        : {results['precision']:.1f}%")
        print(f"  Recall           : {results['recall']:.1f}%")
        print(f"  F1-Score         : {results['f1']:.1f}%")
        print(f"  AUC              : {results['auc']:.4f}")
        print(f"  Results saved to : {args.output}")
        print(f"{'='*50}")

        return 0

    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
