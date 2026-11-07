"""Tests for the v2 pipeline fixes (mocked / synthetic -- no trained model needed).

Covers: sigmoid evaluation (issue #235), confusion-matrix label order,
Grad-CAM class selection for sigmoid models, canonical preprocessing
fallback, API health endpoint and async task cap.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import evaluate
from preprocessing import prepare_model_input


def _write_dataset(root: Path, n: int = 4) -> None:
    for cls, value in (("real", 200), ("fake", 50)):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(d / f"{cls}_{i}.png"), np.full((64, 64, 3), value, np.uint8))


class _SigmoidModel:
    """Returns P(Real) = 0.9 for bright images and 0.1 for dark ones."""

    name = "sigmoid_mock"

    def predict(self, x, verbose=0):
        bright = x.reshape(len(x), -1).mean(axis=1) > 128
        return np.where(bright, 0.9, 0.1).reshape(-1, 1).astype("float32")


def test_evaluate_handles_sigmoid_models(tmp_path):
    _write_dataset(tmp_path / "data")
    out = tmp_path / "metrics.json"
    results = evaluate.run_evaluation(_SigmoidModel(), str(tmp_path / "data"), str(out), batch_size=3)

    assert results["accuracy"] == 100.0
    # rows = true [Fake, Real], cols = predicted [Fake, Real]
    assert results["confusion_matrix"] == [[4, 0], [0, 4]]
    assert results["auc"] == pytest.approx(1.0)
    assert results["class_statistics"]["Real"]["correctly_classified"] == 4
    assert json.loads(out.read_text())["model_version"]


def test_evaluate_sigmoid_misclassification_is_counted(tmp_path):
    _write_dataset(tmp_path / "data")

    class AlwaysReal:
        def predict(self, x, verbose=0):
            return np.full((len(x), 1), 0.8, dtype="float32")

    results = evaluate.run_evaluation(AlwaysReal(), str(tmp_path / "data"), str(tmp_path / "m.json"))
    assert results["accuracy"] == 50.0
    assert results["confusion_matrix"] == [[0, 4], [0, 4]]


def test_real_probability_supports_sigmoid_and_softmax():
    assert evaluate._real_probability(np.array([[0.3], [0.7]])).tolist() == [0.3, 0.7]
    assert evaluate._real_probability(np.array([[0.6, 0.4]])).tolist() == [0.4]
    with pytest.raises(ValueError):
        evaluate._real_probability(np.zeros((2, 3)))


def test_confusion_matrix_plot_labels_match_evaluate_layout():
    from metrics import get_confusion_matrix_plot

    fig = get_confusion_matrix_plot({"confusion_matrix": [[1, 2], [3, 4]]})
    assert list(fig.data[0].x) == ["Fake", "Real"]
    assert list(fig.data[0].y) == ["Fake", "Real"]


def test_gradcam_explains_predicted_class_for_sigmoid():
    tf = pytest.importorskip("tensorflow")
    from gradcam import _class_score

    fake_pred = tf.constant([[0.2]])
    real_pred = tf.constant([[0.9]])
    assert float(_class_score(fake_pred)[0]) == pytest.approx(0.8)  # explains "Fake"
    assert float(_class_score(real_pred)[0]) == pytest.approx(0.9)  # explains "Real"
    assert float(_class_score(tf.constant([[0.1, 0.9]]))[0]) == pytest.approx(0.9)


def test_prepare_model_input_falls_back_to_full_image_when_detector_fails():
    img = np.full((50, 80, 3), 7, np.uint8)

    def broken(_):
        raise RuntimeError("boom")

    batch, face, box = prepare_model_input(img, face_detector=broken)
    assert batch.shape == (1, 96, 96, 3) and batch.dtype == np.float32
    assert box is None and face is img


def test_prepare_model_input_uses_face_crop():
    img = np.zeros((200, 200, 3), np.uint8)
    img[50:150, 50:150] = 255
    batch, face, box = prepare_model_input(img, face_detector=lambda im: (im[50:150, 50:150], (50, 50, 100, 100)))
    assert box == (50, 50, 100, 100)
    assert float(batch.min()) == 255.0  # only the cropped (white) region is fed to the model


def test_decoded_images_are_read_only():
    from preprocessing import decode_image_bytes

    ok, enc = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))
    arr = decode_image_bytes(enc.tobytes())
    with pytest.raises(ValueError):
        arr[0, 0, 0] = 1


def test_health_endpoint_and_task_cap(monkeypatch):
    from fastapi.testclient import TestClient
    import api.main as api_main

    client = TestClient(api_main.app)
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and "model_loaded" in body

    monkeypatch.setattr(api_main, "_validate_image_bytes", lambda _: None)
    monkeypatch.setattr(api_main.task_store, "active_count", lambda: api_main.MAX_ACTIVE_TASKS)
    r = client.post("/api/detect/async", files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 503
