"""REAL-MODEL tests: these load the trained ``deepfake_detection_model.h5``.

They are skipped automatically when the model file is absent.  Tests that
also need labelled face images look in ``PIXELTRUTH_TEST_DATA`` (default
``data/faces/test``, created by ``scripts/prepare_dataset.py``) and are
skipped if it does not exist.  Nothing here is mocked.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.getenv("PIXELTRUTH_MODEL_PATH", ROOT / "deepfake_detection_model.h5"))
TEST_DATA = Path(os.getenv("PIXELTRUTH_TEST_DATA", ROOT / "data" / "faces" / "test"))

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model file not present")


def _sample(cls: str, n: int) -> list[Path]:
    d = TEST_DATA / cls
    if not d.is_dir():
        pytest.skip(f"labelled test images not present at {d}")
    return sorted(d.glob("*.jpg"))[:n]


@pytest.fixture(scope="module")
def model():
    from utils.model_loader import get_model_mtime, load_cached_model

    return load_cached_model(get_model_mtime(str(MODEL)), model_path=str(MODEL))


def test_model_contract(model):
    assert tuple(model.input_shape[1:]) == (96, 96, 3)
    assert model.output_shape[-1] == 1  # sigmoid P(Real)
    assert any(type(l).__name__ == "Rescaling" for l in model.layers)


def test_predict_runs_on_synthetic_image(model):
    import predict

    ok, enc = cv2.imencode(".png", np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8))
    r = predict.predict_image(enc.tobytes(), model_path=str(MODEL))
    assert r["label"] in ("Real", "Fake")
    assert 0.5 <= r["confidence"] <= 1.0


def test_real_model_accuracy_on_sample_faces(model):
    """Smoke check of genuine detection quality on 40 held-out faces."""
    import predict

    correct = 0
    paths = [(p, "Real") for p in _sample("real", 20)] + [(p, "Fake") for p in _sample("fake", 20)]
    for path, truth in paths:
        correct += predict.predict_image(str(path), model_path=str(MODEL))["label"] == truth
    assert correct / len(paths) >= 0.75, f"only {correct}/{len(paths)} correct"


def test_gradcam_on_real_model(model):
    import predict
    from gradcam import find_last_conv_layer, get_backbone_submodel, make_gradcam_heatmap, overlay_heatmap

    path = _sample("fake", 1)[0]
    r = predict.predict_image(str(path), model_path=str(MODEL))
    heat = make_gradcam_heatmap(r["processed_image"], model, find_last_conv_layer(get_backbone_submodel(model)))
    assert heat.ndim == 2 and np.isfinite(heat).all() and heat.max() <= 1.0 + 1e-6
    assert overlay_heatmap(np.ascontiguousarray(r["face_image"]), heat).shape[2] == 3


def test_api_with_real_model():
    from fastapi.testclient import TestClient
    import api.main as api_main

    path = _sample("real", 1)[0]
    with TestClient(api_main.app) as client:  # context manager runs startup (model load)
        assert client.get("/api/health").json()["model_loaded"] is True
        r = client.post("/api/detect", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["verdict"] in ("Real", "Fake") and body["face_detected"] is True

        bad = client.post("/api/detect", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
        assert bad.status_code == 400
