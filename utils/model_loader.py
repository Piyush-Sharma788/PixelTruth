import streamlit as st
import numpy as np

from model_utils import (
    ensure_model_file,
    get_model_path,
    get_model_url,
    get_model_sha256,
)

@st.cache_resource
def load_cached_model(model_path: str | None = None):
    """
    Loads TensorFlow model only once.
    Performs warm-up inference to reduce first prediction latency.
    """

    model_file_path = ensure_model_file(
        model_path=model_path or get_model_path(),
        model_url=get_model_url(),
        model_sha256=get_model_sha256(),
        download_if_missing=True,
    )

    from tensorflow.keras.models import load_model

    model = load_model(model_file_path)

    # Warm-up inference
    dummy_input = np.zeros((1, 96, 96, 3), dtype=np.float32)

    model.predict(dummy_input, verbose=0)

    return model
