from unittest.mock import patch, MagicMock

from utils.model_loader import load_cached_model

@patch("tensorflow.keras.models.load_model")
@patch("utils.model_loader.ensure_model_file")
def test_load_cached_model_memoization(mock_ensure, mock_load_model_func):
    # Clear cache first to ensure a fresh test run
    if hasattr(load_cached_model, "cache_clear"):
        load_cached_model.cache_clear()

    mock_model = MagicMock()
    mock_load_model_func.return_value = mock_model
    mock_ensure.return_value = "dummy_path"

    # First call should execute the function and load the model
    model1 = load_cached_model(model_mtime=1.0, model_path="dummy_path")

    # Second call with the same arguments should return the cached model without reloading
    model2 = load_cached_model(model_mtime=1.0, model_path="dummy_path")

    assert model1 is mock_model
    assert model2 is mock_model
    mock_load_model_func.assert_called_once_with("dummy_path")
    mock_ensure.assert_called_once()
