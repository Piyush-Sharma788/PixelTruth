"""Unit tests for calibration.py temperature scaling module."""
import numpy as np
import pytest

from calibration import temperature_scale


# ─── temperature_scale — binary (sigmoid) output ─────────────────────────────

def test_identity_at_temperature_one():
    """T=1.0 must return probabilities unchanged."""
    probs = np.array([[0.8]])
    result = temperature_scale(probs, temperature=1.0)
    np.testing.assert_allclose(result, probs)


def test_binary_calibration_lowers_high_confidence():
    """Calibration (T>1) should reduce a very high confidence probability."""
    probs = np.array([[0.95]])
    result = temperature_scale(probs, temperature=2.0)
    # calibrated confidence should be lower than raw
    assert float(result.flat[0]) < 0.95


def test_binary_calibration_raises_low_confidence():
    """Calibration (T>1) should raise a very low confidence probability."""
    probs = np.array([[0.05]])
    result = temperature_scale(probs, temperature=2.0)
    # calibrated confidence should be higher than raw (closer to 0.5)
    assert float(result.flat[0]) > 0.05


def test_binary_calibration_preserves_neutral():
    """At p=0.5 (maximum uncertainty) calibration should leave value unchanged."""
    probs = np.array([[0.5]])
    result = temperature_scale(probs, temperature=2.0)
    np.testing.assert_allclose(result, probs, atol=1e-6)


def test_output_shape_preserved_binary():
    """Output shape must match input shape for binary predictions."""
    probs = np.array([[0.8]])
    result = temperature_scale(probs, temperature=1.5)
    assert result.shape == probs.shape


# ─── temperature_scale — multi-class (softmax) output ────────────────────────

def test_softmax_calibration_sums_to_one():
    """Calibrated softmax output must still sum to 1."""
    probs = np.array([[0.1, 0.9]])
    result = temperature_scale(probs, temperature=2.0)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_calibration_lowers_peak_probability():
    """Temperature scaling should reduce the peak (max) class probability."""
    probs = np.array([[0.05, 0.95]])
    result = temperature_scale(probs, temperature=2.0)
    assert float(result[0, 1]) < 0.95


def test_softmax_calibration_preserves_argmax():
    """The winning class should remain the same after calibration."""
    probs = np.array([[0.1, 0.9]])
    result = temperature_scale(probs, temperature=2.0)
    assert np.argmax(result) == np.argmax(probs)


def test_output_shape_preserved_softmax():
    """Output shape must match input shape for softmax predictions."""
    probs = np.array([[0.1, 0.9]])
    result = temperature_scale(probs, temperature=1.5)
    assert result.shape == probs.shape


# ─── temperature_scale — edge cases ──────────────────────────────────────────

def test_invalid_temperature_raises():
    """Zero or negative temperature must raise ValueError."""
    probs = np.array([[0.8]])
    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        temperature_scale(probs, temperature=0.0)
    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        temperature_scale(probs, temperature=-1.0)


# ─── CLI warning integration ──────────────────────────────────────────────────

class FakeModel:
    def __init__(self, output):
        self.output = np.array(output, dtype=np.float32)

    def predict(self, _image, verbose=0):
        return self.output


def make_png_bytes():
    """Create a valid 20×20 black PNG in memory."""
    import cv2
    ok, buf = cv2.imencode(".png", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_cli_no_warning_when_above_threshold(monkeypatch, tmp_path, capsys):
    """No warning printed when calibrated confidence >= threshold."""
    import predict

    monkeypatch.setattr(predict, "load_cached_model", lambda *a, **kw: FakeModel([[0.95]]))
    img = tmp_path / "img.png"
    img.write_bytes(make_png_bytes())

    exit_code = predict.main([str(img), "--temperature", "1.0", "--threshold", "0.70"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[WARNING]" not in captured.err


def test_cli_warning_when_below_threshold(monkeypatch, tmp_path, capsys):
    """[WARNING] is printed to stderr when calibrated confidence < threshold."""
    import predict

    # A score of 0.6 calibrated at T=1.0 gives 60% confidence, below 0.70 threshold
    monkeypatch.setattr(predict, "load_cached_model", lambda *a, **kw: FakeModel([[0.6]]))
    img = tmp_path / "img.png"
    img.write_bytes(make_png_bytes())

    exit_code = predict.main([str(img), "--temperature", "1.0", "--threshold", "0.70"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[WARNING]" in captured.err
    assert "Low-confidence" in captured.err
