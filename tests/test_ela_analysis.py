import numpy as np
from PIL import Image
from io import BytesIO
import pytest

from ela_analysis import compute_ela, ela_uniformity_score

def create_test_image(color=(255, 0, 0), size=(100, 100), format="JPEG"):
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def create_noisy_image(size=(100, 100), format="JPEG"):
    arr = np.random.randint(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def test_compute_ela_returns_same_dimensions():
    img_bytes = create_test_image(size=(120, 80))
    ela = compute_ela(img_bytes)
    assert ela is not None
    assert ela.shape[:2] == (80, 120)

def test_compute_ela_returns_bgr_uint8():
    img_bytes = create_test_image()
    ela = compute_ela(img_bytes)
    assert ela is not None
    assert ela.dtype == np.uint8
    assert ela.ndim == 3
    assert ela.shape[2] == 3

def test_compute_ela_on_uniform_image_is_dark():
    # A solid color image re-saved should have no compression artifacts, meaning diff is 0
    img_bytes = create_test_image(color=(100, 150, 200))
    ela = compute_ela(img_bytes, quality=90)
    assert ela is not None
    # Maximum difference shouldn't be very high
    assert np.max(ela) < 50 

def test_ela_uniformity_score_range():
    img_bytes = create_noisy_image()
    ela = compute_ela(img_bytes)
    score = ela_uniformity_score(ela)
    assert 0.0 <= score <= 1.0

def test_compute_ela_handles_png_input():
    img_bytes = create_test_image(format="PNG")
    ela = compute_ela(img_bytes)
    assert ela is not None

def test_compute_ela_returns_none_for_garbage():
    garbage = b"not an image at all"
    ela = compute_ela(garbage)
    assert ela is None

def test_ela_uniformity_score_uniform_vs_noisy():
    # Uniform image
    uniform_bytes = create_test_image(color=(128, 128, 128))
    uniform_ela = compute_ela(uniform_bytes)
    uniform_score = ela_uniformity_score(uniform_ela)
    
    # Noisy image
    noisy_bytes = create_noisy_image()
    noisy_ela = compute_ela(noisy_bytes)
    noisy_score = ela_uniformity_score(noisy_ela)
    
    # Uniform ELA should have higher score (more uniform)
    assert uniform_score > noisy_score
