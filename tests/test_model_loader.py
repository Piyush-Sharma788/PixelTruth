import sys
from unittest.mock import patch, MagicMock

# Mock tensorflow modules to prevent ModuleNotFoundError
mock_tf = MagicMock()
mock_keras = MagicMock()
mock_models = MagicMock()
mock_tf.keras = mock_keras
mock_keras.models = mock_models
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_keras
sys.modules["tensorflow.keras.models"] = mock_models

# Mock load_model function
mock_load_model_func = MagicMock()
mock_models.load_model = mock_load_model_func

from utils.model_loader import load_cached_model


def test_load_cached_model_memoization():
    # Clear cache first to ensure a fresh test run
    if hasattr(load_cached_model, "cache_clear"):
        load_cached_model.cache_clear()

    mock_model = MagicMock()
    mock_load_model_func.return_value = mock_model

    with patch(
        "utils.model_loader.ensure_model_file", return_value="dummy_path"
    ) as mock_ensure:

        # First call should execute the function and load the model
        model1 = load_cached_model(model_mtime=1.0, model_path="dummy_path")

        # Second call with the same arguments should return the cached model without reloading
        model2 = load_cached_model(model_mtime=1.0, model_path="dummy_path")

        assert model1 is mock_model
        assert model2 is mock_model
        mock_load_model_func.assert_called_once_with("dummy_path")
        mock_ensure.assert_called_once()
