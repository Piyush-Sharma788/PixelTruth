"""Thread-safe, framework-agnostic model loader.

This module provides a singleton TensorFlow model that is safe to use
from FastAPI async workers, CLI scripts, and Streamlit simultaneously.

Design decisions
----------------
* **No Streamlit dependency** — `@st.cache_resource` is a Streamlit-specific
  primitive tied to the script-rerun execution model; it is not thread-safe
  across asyncio worker threads (CWE-1048/CWE-1061).  We use a plain
  `threading.Lock` + module-level variable instead.
* **Warm-up inference** — a dummy forward pass is performed immediately after
  loading to reduce first-request latency.
* **mtime-based invalidation** — callers may pass `model_mtime` to trigger a
  reload when the model file changes on disk (useful during development).
"""

from __future__ import annotations

import threading
import numpy as np
import os

from model_utils import (
    ensure_model_file,
    get_model_path,
    get_model_url,
    get_model_sha256,
)

MODEL_PATH = get_model_path()
MODEL_URL = get_model_url()
MODEL_SHA256 = get_model_sha256()

# ---------------------------------------------------------------------------
# Module-level singleton state — protected by _model_lock.
# ---------------------------------------------------------------------------
_model_lock: threading.Lock = threading.Lock()
_cached_model = None
_cached_mtime: float | None = None


def get_model_mtime(model_path: str | None = None) -> float:
    """Return the last-modified time of the model file, or 0.0 on error."""
    try:
        return os.path.getmtime(model_path or MODEL_PATH)
    except OSError:
        return 0.0


def load_cached_model(model_mtime: float | None = None, model_path: str | None = None):
    """Return the singleton TensorFlow model, loading or reloading as needed.

    Parameters
    ----------
    model_mtime:
        If provided and different from the cached mtime, the model is reloaded
        from disk.  Pass ``get_model_mtime()`` to enable hot-reload on file
        change.
    model_path:
        Override the model file path (defaults to ``MODEL_PATH`` / env var).

    Returns
    -------
    tensorflow.keras.Model
        The loaded (and warmed-up) model instance.

    Notes
    -----
    This function is safe to call from:
    * FastAPI / Uvicorn async worker threads (no Streamlit context required).
    * CLI scripts (``predict.py``).
    * Streamlit UI (``app.py``) — Streamlit should use its own
      ``@st.cache_resource`` wrapper in ``app.py`` if per-session caching is
      desired, rather than delegating that concern to this shared module.
    """
    global _cached_model, _cached_mtime

    resolved_path = model_path or MODEL_PATH
    resolved_mtime = model_mtime if model_mtime is not None else get_model_mtime(resolved_path)

    with _model_lock:
        if _cached_model is None or _cached_mtime != resolved_mtime:
            model_file = ensure_model_file(
                model_path=resolved_path,
                model_url=get_model_url(),
                model_sha256=get_model_sha256(),
                download_if_missing=True,
            )

            from tensorflow.keras.models import load_model  # deferred import

            model = load_model(model_file)

            # Warm-up inference to reduce first-prediction latency.
            dummy_input = np.zeros((1, 96, 96, 3), dtype=np.float32)
            model.predict(dummy_input, verbose=0)

            _cached_model = model
            _cached_mtime = resolved_mtime

    return _cached_model
