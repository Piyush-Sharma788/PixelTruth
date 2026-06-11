import functools
import logging

import numpy as np

from config import HF_MODEL_NAME

logger = logging.getLogger(__name__)


def _memoize_cache_resource(func):
    return functools.lru_cache(maxsize=1)(func)


try:
    import streamlit as st
    if st is not None and hasattr(st, "runtime") and st.runtime.exists():
        cache_resource = st.cache_resource
    else:
        cache_resource = _memoize_cache_resource
except (ImportError, AttributeError):
    cache_resource = _memoize_cache_resource


_HF_PROCESSOR = None
_HF_MODEL = None


def get_hf_model():
    global _HF_PROCESSOR, _HF_MODEL
    if _HF_MODEL is not None:
        return _HF_PROCESSOR, _HF_MODEL

    from transformers import AutoImageProcessor, SiglipForImageClassification

    logger.info("Loading Hugging Face model: %s", HF_MODEL_NAME)
    _HF_PROCESSOR = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    _HF_MODEL = SiglipForImageClassification.from_pretrained(HF_MODEL_NAME)
    _HF_MODEL.eval()
    logger.info("Hugging Face model loaded successfully")
    return _HF_PROCESSOR, _HF_MODEL


@cache_resource
def load_cached_model():
    processor, model = get_hf_model()
    dummy_input = processor(
        images=np.zeros((224, 224, 3), dtype=np.uint8),
        return_tensors="pt",
    )
    model(**dummy_input)
    return processor, model
