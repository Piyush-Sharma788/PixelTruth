import numpy as np
import pytest

import inference
import predict


def test_inference_decode_prediction_uses_canonical_decoder(monkeypatch):
    seen = {}

    def fake_decode(prediction):
        seen["shape"] = np.asarray(prediction).shape
        return "Real", 0.91, [0.09, 0.91]

    monkeypatch.setattr(predict, "decode_prediction", fake_decode)

    label, confidence = inference.decode_prediction(np.array([[0.09, 0.91]], dtype=np.float32))

    assert seen["shape"] == (1, 2)
    assert label == "Real"
    assert confidence == pytest.approx(0.91)
