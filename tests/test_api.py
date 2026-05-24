from fastapi.testclient import TestClient

import api.main as api_main


def test_api_rejects_non_image_upload():
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.txt", b"data", "text/plain")}
    )

    assert response.status_code == 400


def test_api_returns_prediction(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes: {"label": "Real", "confidence": 0.8, "raw": [0.8]},
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
