from fastapi.testclient import TestClient
import pytest

import api.main as api_main


def _fake_predict(image_bytes):
    return {"label": "Real", "confidence": 0.8, "raw": [0.8]}


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
        "face_detected": False,
        "face_box": None,
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


# ---------------------------------------------------------------------------
# API key authentication tests
# ---------------------------------------------------------------------------


def test_api_key_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "test-secret-key")
    monkeypatch.setattr(api_main, "predict_image", _fake_predict)
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("img.png", b"data", "image/png")}
    )

    assert response.status_code == 401
    assert "API key" in response.json()["detail"]


def test_api_key_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "test-secret-key")
    monkeypatch.setattr(api_main, "predict_image", _fake_predict)
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect",
        files={"file": ("img.png", b"data", "image/png")},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_api_key_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "test-secret-key")
    monkeypatch.setattr(api_main, "predict_image", _fake_predict)
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect",
        files={"file": ("img.png", b"data", "image/png")},
        headers={"X-API-Key": "test-secret-key"},
    )

    assert response.status_code == 200


def test_api_key_skipped_when_unset(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(api_main, "predict_image", _fake_predict)
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("img.png", b"data", "image/png")}
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# CORS configuration tests
# ---------------------------------------------------------------------------


def test_cors_allows_configured_origin():
    client = TestClient(api_main.app)

    response = client.options(
        "/api/detect",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_rejects_unknown_origin():
    client = TestClient(api_main.app)

    response = client.options(
        "/api/detect",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(api_main, "predict_image", _fake_predict)

    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from fastapi import FastAPI, UploadFile, File, Request

    test_limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["2/minute"],
    )

    test_app = FastAPI()
    test_app.state.limiter = test_limiter
    test_app.add_exception_handler(
        RateLimitExceeded, api_main._rate_limit_exceeded_handler
    )

    @test_app.post("/api/detect")
    @test_limiter.limit("2/minute")
    async def detect(request: Request, file: UploadFile = File(...)):
        return {"verdict": "Real", "confidence": 0.8, "raw_scores": [0.8]}

    client = TestClient(test_app)

    for _ in range(2):
        resp = client.post(
            "/api/detect", files={"file": ("img.png", b"data", "image/png")}
        )
        assert resp.status_code == 200

    resp = client.post(
        "/api/detect", files={"file": ("img.png", b"data", "image/png")}
    )

    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
