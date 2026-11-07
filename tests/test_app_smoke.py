"""Headless Streamlit smoke test: app.py must start and switch modes without
raising. Uses the bundled model when present (real model), otherwise checks
that the app degrades to an error message instead of crashing."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PIXELTRUTH_DB_PATH", str(tmp_path / "history.db"))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_app_starts_without_exceptions(app):
    at = app.run()
    assert not at.exception, [e.value for e in at.exception]
    model_present = (ROOT / "deepfake_detection_model.h5").exists()
    if model_present:
        assert any("Model initialized" in s.value for s in at.success)
    else:
        assert any("Error loading model" in e.value for e in at.error)
    # No hard-coded accuracy claims: the snapshot must come from metrics_cache.json
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("Test Accuracy") != "95%"


def test_app_switches_to_forensic_comparison(app):
    at = app.run()
    at.radio[0].set_value("Forensic Comparison").run()
    assert not at.exception, [e.value for e in at.exception]
