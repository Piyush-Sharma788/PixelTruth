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
        return {
            "label": "Real",
            "confidence": 0.8,
            "raw": [0.8],
            "face_detected": True,
            "face_box": (10, 20, 30, 40),
        }

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
        "face_detected": True,
        "face_box": [10, 20, 30, 40],
    }
    assert provided_inputs == [b"data"]


def test_api_returns_prediction_without_face_metadata(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes: {"label": "Fake", "confidence": 0.2, "raw": [0.2]},
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "verdict": "Fake",
        "confidence": 0.2,
        "raw_scores": [0.2],
        "face_detected": False,
        "face_box": None,
    }


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


def test_async_detect_returns_task_id(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes: {
            "label": "Real",
            "confidence": 0.95,
            "raw": [0.95],
            "face_detected": True,
            "face_box": (1, 2, 3, 4),
        },
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect/async", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 202
    assert "task_id" in response.json()
    assert isinstance(response.json()["task_id"], str)


def test_task_status_returns_completed(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes: {
            "label": "Fake",
            "confidence": 0.15,
            "raw": [0.15],
            "face_detected": True,
            "face_box": (5, 6, 7, 8),
        },
    )
    client = TestClient(api_main.app)

    submit_response = client.post(
        "/api/detect/async", files={"file": ("sample.png", b"data", "image/png")}
    )
    task_id = submit_response.json()["task_id"]

    status_response = client.get(f"/api/task/{task_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["verdict"] == "Fake"
    assert status_response.json()["confidence"] == 0.15
    assert status_response.json()["raw_scores"] == [0.15]
    assert status_response.json()["face_detected"] is True
    assert status_response.json()["face_box"] == [5, 6, 7, 8]


def test_task_not_found_returns_404():
    client = TestClient(api_main.app)

    response = client.get("/api/task/nonexistent")

    assert response.status_code == 404


def test_async_detect_rejects_non_image(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes: pytest.fail("prediction should not run for invalid async uploads"),
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect/async", files={"file": ("sample.txt", b"data", "text/plain")}
    )

    assert response.status_code == 400
