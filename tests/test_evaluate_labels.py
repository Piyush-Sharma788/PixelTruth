"""Regression test: evaluate.py class-label mapping must match training.

The Keras training pipeline (train_v3.py) uses
``image_dataset_from_directory`` with ``label_mode="binary"``, which
assigns classes **alphabetically**: ``fake/ → 0``, ``real/ → 1``.

``evaluate.py::_collect_image_paths`` must use the same mapping so
that ground-truth labels align with model predictions.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluate import _collect_image_paths


@pytest.fixture()
def test_dataset(tmp_path: Path):
    """Create a minimal test dataset with one image per class."""
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    # Create tiny valid PNG files (1x1 pixel)
    import cv2

    for name, directory in [("real_001.png", real_dir), ("fake_001.png", fake_dir)]:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(directory / name), img)

    return tmp_path


class TestCollectImagePathsLabelMapping:
    """Verify _collect_image_paths assigns labels matching the Keras
    alphabetical convention: fake/ → 0, real/ → 1."""

    @staticmethod
    def _is_in_subdir(path: str, subdir_name: str) -> bool:
        """Check if *path* is under a directory component named *subdir_name*."""
        return os.sep + subdir_name + os.sep in path or path.startswith(subdir_name + os.sep)

    def test_real_images_are_labelled_1(self, test_dataset):
        paths, labels = _collect_image_paths(str(test_dataset))
        real_indices = [i for i, p in enumerate(paths) if self._is_in_subdir(p, "real")]
        assert real_indices, "No real images found in paths"
        for idx in real_indices:
            assert labels[idx] == 1, (
                f"Real image at index {idx} has label {labels[idx]}; "
                f"expected 1 (Keras alphabetical: fake=0, real=1)"
            )

    def test_fake_images_are_labelled_0(self, test_dataset):
        paths, labels = _collect_image_paths(str(test_dataset))
        fake_indices = [i for i, p in enumerate(paths) if self._is_in_subdir(p, "fake")]
        assert fake_indices, "No fake images found in paths"
        for idx in fake_indices:
            assert labels[idx] == 0, (
                f"Fake image at index {idx} has label {labels[idx]}; "
                f"expected 0 (Keras alphabetical: fake=0, real=1)"
            )

    def test_labels_match_training_convention(self, test_dataset):
        """End-to-end: the label set {0, 1} must map to {Fake, Real}
        in the same way as predict.py::decode_prediction."""
        paths, labels = _collect_image_paths(str(test_dataset))

        label_map = {}
        for path, label in zip(paths, labels):
            if self._is_in_subdir(path, "real"):
                label_map["Real"] = label
            elif self._is_in_subdir(path, "fake"):
                label_map["Fake"] = label

        # predict.py: "Real" if class_index == 1 else "Fake"
        assert label_map["Real"] == 1, "Real must be class 1 (matching predict.py)"
        assert label_map["Fake"] == 0, "Fake must be class 0 (matching predict.py)"
