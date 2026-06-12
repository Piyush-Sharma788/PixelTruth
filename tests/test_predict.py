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


def test_predict_image_calls_face_detection_for_ndarray_input(monkeypatch):
    import numpy as np
    from preprocessing import preprocess_image_array

    detected_faces = []

    def fake_detect_and_crop_face(image):
        detected_faces.append(True)
        return image, (10, 10, 50, 50)

    monkeypatch.setattr(
        "predict.detect_and_crop_face", fake_detect_and_crop_face
    )
    monkeypatch.setattr(
        predict, "load_cached_model", lambda *args, **kwargs: FakeModel([[0.8]])
    )

    bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
    predict.predict_image(bgr_image)

    assert len(detected_faces) == 1, (
        "predict_image() must call detect_and_crop_face for numpy array input"
    )


def test_predict_image_falls_back_to_full_image_when_face_detection_fails(monkeypatch):
    import numpy as np

    def failing_detect(image):
        raise RuntimeError("Face detection crashed")

    monkeypatch.setattr(
        "predict.detect_and_crop_face", failing_detect
    )
    monkeypatch.setattr(
        predict, "load_cached_model", lambda *args, **kwargs: FakeModel([[0.8]])
    )

    bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = predict.predict_image(bgr_image)

    assert result["label"] in ("Real", "Fake")
    assert result["face_detected"] is False or result["face_detected"] is None


def test_decode_prediction_rejects_unknown_output_shape():
    with pytest.raises(ModelExecutionError, match="Unsupported model output shape"):
        predict.decode_prediction(np.array([[0.1, 0.2, 0.7]]))


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
