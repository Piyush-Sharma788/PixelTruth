import numpy as np
import pytest

from inference import predict_image


class FakeModel:
    def __init__(self, score):
        self.score = score

    def predict(self, _processed, verbose=0):
        return np.array([[self.score]], dtype=np.float32)


@pytest.mark.parametrize(
    ("score", "label", "confidence"),
    [(0.9, "Real", 0.9), (0.1, "Fake", 0.9)],
)
def test_streamlit_inference_decodes_sigmoid_predictions(score, label, confidence):
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    result_label, result_confidence, processed = predict_image(
        FakeModel(score), image
    )

    assert result_label == label
    assert result_confidence == pytest.approx(confidence)
    assert processed.shape == (1, 96, 96, 3)
