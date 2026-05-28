from fastapi.testclient import TestClient
import pytest

import api.main as api_main


def test_api_rejects_non_image_upload():
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.txt", b"data", "text/plain")}
    )

    assert response.status_code == 400


def test_api_returns_prediction(monkeypatch):
    provided_inputs = []

    def fake_predict(image_bytes):
        provided_inputs.append(image_bytes)
        return {"label": "Real", "confidence": 0.8, "raw": [0.8]}

    monkeypatch.setattr(
        api_main,
        "predict_image",
        fake_predict,
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "verdict": "Real",
        "confidence": 0.8,
        "raw_scores": [0.8],
    }
    assert provided_inputs == [b"data"]


def test_api_rejects_oversized_upload_before_prediction(monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_SIZE_BYTES", 3)
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes: pytest.fail("prediction should not run for oversized input"),
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 413
