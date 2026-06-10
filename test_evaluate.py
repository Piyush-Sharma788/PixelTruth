"""
Unit tests for the evaluation harness (evaluate.py) and
refactored metrics module (metrics.py).

Run with::

    pytest test_evaluate.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import evaluate
import metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Return a random BGR image as a numpy array."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _save_image(directory: str, filename: str, image: np.ndarray) -> str:
    """Write an image to disk and return the full path."""
    path = os.path.join(directory, filename)
    cv2.imwrite(path, image)
    return path


def _make_test_dataset(base_dir: str, n_real: int = 5, n_fake: int = 5):
    """Create a test dataset directory with real/ and fake/ subdirs."""
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i in range(n_real):
        _save_image(real_dir, f"real_{i:03d}.png", _make_test_image())

    for i in range(n_fake):
        _save_image(fake_dir, f"fake_{i:03d}.png", _make_test_image())


def _make_mock_model(n_classes: int = 2) -> MagicMock:
    """Return a mock Keras model that produces plausible softmax outputs."""
    mock = MagicMock()

    def fake_predict(x, **kwargs):
        batch_size = x.shape[0]
        # Generate random softmax-like outputs
        raw = np.random.rand(batch_size, n_classes)
        return raw / raw.sum(axis=1, keepdims=True)

    mock.predict = MagicMock(side_effect=fake_predict)
    mock.name = "test_model"
    return mock


def _make_sample_cache() -> dict:
    """Return a valid metrics cache dict for testing."""
    return {
        "accuracy": 95.2,
        "precision": 94.8,
        "recall": 95.7,
        "f1": 95.2,
        "auc": 0.983,
        "confusion_matrix": [[475, 25], [23, 477]],
        "roc": {
            "fpr": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            "tpr": [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0],
        },
        "class_statistics": {
            "Real": {
                "total_samples": 500,
                "correctly_classified": 475,
                "misclassified": 25,
                "class_accuracy": 95.0,
            },
            "Fake": {
                "total_samples": 500,
                "correctly_classified": 477,
                "misclassified": 23,
                "class_accuracy": 95.4,
            },
        },
        "dataset_distribution": {"Real": 500, "Fake": 500},
        "evaluated_at": "2026-05-25T16:30:00+00:00",
        "model_path": "test_model",
        "test_data_dir": "test_data",
        "total_images": 1000,
        "skipped_images": 0,
    }


# ===========================================================================
# evaluate.py — _collect_image_paths
# ===========================================================================

class TestCollectImagePaths:

    def test_collects_real_and_fake_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=3, n_fake=4)
            paths, labels = evaluate._collect_image_paths(tmpdir)

            assert len(paths) == 7
            assert len(labels) == 7
            assert labels.count(0) == 3  # Real
            assert labels.count(1) == 4  # Fake

    def test_raises_when_dir_missing(self):
        with pytest.raises(FileNotFoundError, match="Test directory not found"):
            evaluate._collect_image_paths("/nonexistent/path")

    def test_raises_when_real_subdir_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "fake"))
            _save_image(os.path.join(tmpdir, "fake"), "img.png", _make_test_image())
            with pytest.raises(FileNotFoundError, match="Missing 'real/'"):
                evaluate._collect_image_paths(tmpdir)

    def test_raises_when_fake_subdir_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "real"))
            _save_image(os.path.join(tmpdir, "real"), "img.png", _make_test_image())
            with pytest.raises(FileNotFoundError, match="Missing 'fake/'"):
                evaluate._collect_image_paths(tmpdir)

    def test_raises_when_class_dir_has_no_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "real"))
            os.makedirs(os.path.join(tmpdir, "fake"))
            _save_image(os.path.join(tmpdir, "real"), "img.png", _make_test_image())
            # fake/ is empty
            with pytest.raises(ValueError, match="No supported images found"):
                evaluate._collect_image_paths(tmpdir)

    def test_ignores_non_image_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=2, n_fake=2)
            # Add a non-image file
            with open(os.path.join(tmpdir, "real", "readme.txt"), "w") as f:
                f.write("not an image")
            paths, labels = evaluate._collect_image_paths(tmpdir)
            assert len(paths) == 4  # Only image files


# ===========================================================================
# evaluate.py — run_evaluation
# ===========================================================================

class TestRunEvaluation:

    def test_produces_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=5, n_fake=5)
            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            results = evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
                batch_size=4,
            )

            assert os.path.isfile(output_path)

            with open(output_path, "r") as f:
                loaded = json.load(f)

            # Verify all required keys
            required_keys = {
                "accuracy", "precision", "recall", "f1", "auc",
                "confusion_matrix", "roc", "class_statistics",
                "dataset_distribution", "evaluated_at",
                "total_images", "skipped_images",
            }
            assert required_keys.issubset(set(loaded.keys()))

    def test_results_have_correct_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=3, n_fake=3)
            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            results = evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
            )

            assert isinstance(results["accuracy"], float)
            assert isinstance(results["precision"], float)
            assert isinstance(results["recall"], float)
            assert isinstance(results["f1"], float)
            assert isinstance(results["auc"], float)
            assert isinstance(results["confusion_matrix"], list)
            assert len(results["confusion_matrix"]) == 2
            assert len(results["confusion_matrix"][0]) == 2
            assert isinstance(results["roc"]["fpr"], list)
            assert isinstance(results["roc"]["tpr"], list)
            assert isinstance(results["total_images"], int)

    def test_confusion_matrix_sums_to_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=4, n_fake=4)
            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            results = evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
            )

            cm = np.array(results["confusion_matrix"])
            assert cm.sum() == results["total_images"]

    def test_dataset_distribution_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=6, n_fake=4)
            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            results = evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
            )

            assert results["dataset_distribution"]["Real"] == 6
            assert results["dataset_distribution"]["Fake"] == 4

    def test_roc_fpr_tpr_same_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=5, n_fake=5)
            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            results = evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
            )

            assert len(results["roc"]["fpr"]) == len(results["roc"]["tpr"])
            assert len(results["roc"]["fpr"]) > 0

    def test_evaluated_at_is_iso_timestamp(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=3, n_fake=3)
            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            results = evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
            )

            # Should parse without error
            datetime.fromisoformat(results["evaluated_at"])

    def test_skips_corrupt_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=3, n_fake=3)

            # Add a corrupt file
            corrupt_path = os.path.join(tmpdir, "real", "corrupt.png")
            with open(corrupt_path, "wb") as f:
                f.write(b"not a valid image at all")

            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            results = evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
            )

            assert results["skipped_images"] >= 1
            assert results["total_images"] == 6  # Only 6 valid images


# ===========================================================================
# evaluate.py — CLI parser
# ===========================================================================

class TestEvaluateCLI:

    def test_parser_requires_test_dir(self):
        parser = evaluate.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing required --test-dir

    def test_parser_accepts_all_args(self):
        parser = evaluate.build_parser()
        args = parser.parse_args([
            "--test-dir", "path/to/test",
            "--output", "custom.json",
            "--model", "my_model.h5",
            "--batch-size", "64",
        ])
        assert args.test_dir == "path/to/test"
        assert args.output == "custom.json"
        assert args.model == "my_model.h5"
        assert args.batch_size == 64

    def test_parser_defaults(self):
        parser = evaluate.build_parser()
        args = parser.parse_args(["--test-dir", "data"])
        assert args.output == "metrics_cache.json"
        assert args.model is None
        assert args.batch_size == 32


# ===========================================================================
# metrics.py — load_cached_metrics
# ===========================================================================

class TestLoadCachedMetrics:

    def test_returns_none_when_file_missing(self):
        result = metrics.load_cached_metrics("/nonexistent/path.json")
        assert result is None

    def test_loads_valid_cache(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(_make_sample_cache(), f)
            path = f.name

        try:
            result = metrics.load_cached_metrics(path)
            assert result is not None
            assert result["accuracy"] == 95.2
        finally:
            os.unlink(path)

    def test_returns_none_for_malformed_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{invalid json")
            path = f.name

        try:
            result = metrics.load_cached_metrics(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_returns_none_when_required_keys_missing(self):
        incomplete_data = {"accuracy": 95.0}  # Missing other required keys

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(incomplete_data, f)
            path = f.name

        try:
            result = metrics.load_cached_metrics(path)
            assert result is None
        finally:
            os.unlink(path)


# ===========================================================================
# metrics.py — data-source functions with None input
# ===========================================================================

class TestMetricsNoneInput:
    """All data-source functions should return None when no cache is available."""

    def test_get_sample_metrics_returns_none(self):
        assert metrics.get_sample_metrics(None) is None

    def test_get_confusion_matrix_plot_returns_none(self):
        assert metrics.get_confusion_matrix_plot(None) is None

    def test_get_roc_curve_plot_returns_none(self):
        assert metrics.get_roc_curve_plot(None) is None

    def test_get_dataset_distribution_plot_returns_none(self):
        assert metrics.get_dataset_distribution_plot(None) is None

    def test_get_class_statistics_returns_none(self):
        assert metrics.get_class_statistics(None) is None

    def test_get_evaluated_at_returns_none(self):
        assert metrics.get_evaluated_at(None) is None

    def test_get_total_images_returns_none(self):
        assert metrics.get_total_images(None) is None


# ===========================================================================
# metrics.py — data-source functions with valid cache
# ===========================================================================

class TestMetricsWithCache:
    """Data-source functions should return correct types with valid cache."""

    @pytest.fixture
    def cache(self):
        return _make_sample_cache()

    def test_get_sample_metrics_returns_dict(self, cache):
        result = metrics.get_sample_metrics(cache)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert result["accuracy"] == 95.2

    def test_get_confusion_matrix_plot_returns_figure(self, cache):
        import plotly.graph_objects as go
        fig = metrics.get_confusion_matrix_plot(cache)
        assert isinstance(fig, go.Figure)

    def test_get_roc_curve_plot_returns_figure(self, cache):
        import plotly.graph_objects as go
        fig = metrics.get_roc_curve_plot(cache)
        assert isinstance(fig, go.Figure)

    def test_get_dataset_distribution_plot_returns_figure(self, cache):
        import plotly.graph_objects as go
        fig = metrics.get_dataset_distribution_plot(cache)
        assert isinstance(fig, go.Figure)

    def test_get_class_statistics_returns_dict(self, cache):
        result = metrics.get_class_statistics(cache)
        assert isinstance(result, dict)
        assert "Real" in result
        assert "Fake" in result
        assert result["Real"]["total_samples"] == 500

    def test_get_evaluated_at_returns_string(self, cache):
        result = metrics.get_evaluated_at(cache)
        assert isinstance(result, str)
        assert "2026" in result

    def test_get_total_images_returns_int(self, cache):
        result = metrics.get_total_images(cache)
        assert result == 1000


# ===========================================================================
# Integration: evaluate.py output is loadable by metrics.py
# ===========================================================================

class TestEvaluateMetricsIntegration:
    """Verify that evaluate.py output can be consumed by metrics.py."""

    def test_evaluation_output_loadable_by_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_test_dataset(tmpdir, n_real=5, n_fake=5)
            output_path = os.path.join(tmpdir, "test_metrics.json")
            model = _make_mock_model()

            evaluate.run_evaluation(
                model=model,
                test_data_dir=tmpdir,
                output_path=output_path,
            )

            # Load with metrics.py
            cached = metrics.load_cached_metrics(output_path)
            assert cached is not None

            # All data-source functions should work
            assert metrics.get_sample_metrics(cached) is not None
            assert metrics.get_confusion_matrix_plot(cached) is not None
            assert metrics.get_roc_curve_plot(cached) is not None
            assert metrics.get_dataset_distribution_plot(cached) is not None
            assert metrics.get_class_statistics(cached) is not None
            assert metrics.get_evaluated_at(cached) is not None
            assert metrics.get_total_images(cached) is not None
