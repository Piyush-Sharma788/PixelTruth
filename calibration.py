"""Confidence calibration utilities for deep learning predictions."""

import numpy as np

def temperature_scale(probabilities: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    """
    Apply temperature scaling to raw confidence probabilities or logits.
    
    If probabilities is a 1D/2D array representing a binary probability (e.g., probability of class 1),
    we convert it to log-odds (logit), divide by temperature, and apply the sigmoid function.
    
    If probabilities represents multi-class softmax scores, we take the log of probabilities,
    divide by temperature, and re-apply softmax.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive.")
        
    if temperature == 1.0:
        return probabilities

    # Avoid divide-by-zero or log of zero by clipping probabilities
    eps = 1e-15
    probs = np.clip(probabilities, eps, 1.0 - eps)

    # If it's a binary probability output (e.g. sigmoid)
    # Binary outputs can have shape (N, 1) or just (1,) or (N,)
    shape = probs.shape
    flat_probs = probs.reshape(-1)
    
    # Check if this is a binary probability (usually shape is (N, 1) or size is 1)
    # The config and model output size == 1 handles binary.
    # If the last dimension is 1, it's binary.
    if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1) or (len(shape) == 2 and shape[1] == 1) or shape == (1,):
        # Convert to logits: log(p / (1-p))
        logits = np.log(flat_probs / (1.0 - flat_probs))
        # Scale by temperature
        scaled_logits = logits / temperature
        # Sigmoid
        scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        return scaled_probs.reshape(shape)

    # If it's a multi-class softmax probability distribution (usually last dim >= 2)
    elif shape[-1] >= 2:
        # Convert to logits: log(p)
        logits = np.log(probs)
        # Scale by temperature
        scaled_logits = logits / temperature
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return probabilities
