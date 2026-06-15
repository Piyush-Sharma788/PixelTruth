import json

import cv2
import numpy as np
import pytest

import predict
from exceptions import ModelExecutionError


class FakeModel:
    def __init__(self, output):
        self.output = np.array(output, dtype=np.float32)

    def predict(self, _image, verbose=0):
        assert verbose == 0
        return self.output


def image_bytes():
    ok, encoded = cv2.imencode(".png", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize(
    ("score", "label", "confidence"),
    [(0.8, "Real", 0.8), (0.2, "Fake", 0.8)],
)
def test_predict_image_decodes_sigmoid_output(monkeypatch, score, label, confidence):
    monkeypatch.setattr(
        predict, "load_cached_model", lambda *args, **kwargs: FakeModel([[score]])
    )

    result = predict.predict_image(image_bytes())

    assert result["label"] == label
    assert result["confidence"] == pytest.approx(confidence)
    assert result["processed_image"].shape == (1, 96, 96, 3)


def test_predict_image_passes_model_override_and_supports_softmax(monkeypatch):
    requested_paths = []

    def load_model(model_mtime=None, model_path=None):
        requested_paths.append(model_path)
        return FakeModel([[0.1, 0.9]])

    monkeypatch.setattr(predict, "load_cached_model", load_model)

    result = predict.predict_image(image_bytes(), model_path="selected.h5")

    assert requested_paths == ["selected.h5"]
    assert result["label"] == "Real"
    assert result["confidence"] == pytest.approx(0.9)


def test_decode_prediction_rejects_unknown_output_shape():
    with pytest.raises(ModelExecutionError, match="Unsupported model output shape"):
        predict.decode_prediction(np.array([[0.1, 0.2, 0.7]]))


def test_cli_accepts_image_path_and_returns_prediction(
    monkeypatch, tmp_path, capsys
):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(image_bytes())
    monkeypatch.setattr(
        predict, "load_cached_model", lambda *args, **kwargs: FakeModel([[0.8]])
    )

    exit_code = predict.main([str(image_path)])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Prediction" in captured
    assert "Confidence" in captured


def test_cli_rejects_nonexistent_path(capsys):
    exit_code = predict.main(["/nonexistent/image.png"])
    assert exit_code == 1
    captured = capsys.readouterr().err
    assert "FileNotFoundError" in captured or "No such file" in captured or "not found" in captured


def test_cli_json_does_not_attempt_to_serialize_processed_image(
    monkeypatch, tmp_path, capsys
):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(image_bytes())
    monkeypatch.setattr(
        predict, "load_cached_model", lambda *args, **kwargs: FakeModel([[0.8]])
    )

    assert predict.main([str(image_path), "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["label"] == "Real"
    assert "processed_image" not in output
