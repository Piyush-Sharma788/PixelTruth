"""Error Level Analysis for deepfake image inspection."""
from __future__ import annotations
from io import BytesIO
import numpy as np
from PIL import Image
import cv2

def compute_ela(image_bytes: bytes, quality: int = 90, amplifier: int = 15) -> np.ndarray:
    """
    Run ELA on raw image bytes.
    
    Args:
      image_bytes: Raw bytes of the original image.
      quality:     JPEG re-save quality (default 90).
      amplifier:   Multiplier applied to the abs difference (default 15).
    
    Returns:
      BGR numpy array (uint8) of the amplified difference image, 
      same dimensions as input. Returns None if ELA cannot be computed
      (e.g. corrupt input).
    """
    try:
        # 1. Open with PIL from BytesIO
        original = Image.open(BytesIO(image_bytes))
        
        # 2. Convert to RGB (handles RGBA, palette modes)
        original = original.convert("RGB")
        
        # 3. Save to an in-memory BytesIO buffer at the given JPEG quality
        buffer = BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        
        # 4. Re-open the saved buffer
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        # 5. Convert both original and re-saved to float32 numpy arrays
        orig_arr = np.array(original, dtype=np.float32)
        resaved_arr = np.array(resaved, dtype=np.float32)
        
        # 6. Compute abs(original - resaved) * amplifier
        diff = np.abs(orig_arr - resaved_arr) * amplifier
        
        # 7. Clip to [0, 255], cast to uint8
        diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)
        
        # 8. Convert RGB -> BGR for OpenCV/Streamlit display consistency
        bgr_diff = diff_clipped[..., ::-1].copy()
        
        return bgr_diff
    except Exception:
        return None

def ela_uniformity_score(ela_image: np.ndarray) -> float:
    """
    Returns a score in [0, 1] where higher = more uniform ELA = 
    more likely AI-generated.
    
    Implementation: compute std deviation of the grayscale ELA image.
    Low std = uniform = suspicious. Normalise: score = 1 - (std / 128).
    Clip to [0, 1].
    """
    if ela_image is None or ela_image.size == 0:
        return 0.0
        
    # Convert BGR to Grayscale for std dev calculation
    if len(ela_image.shape) == 3 and ela_image.shape[2] == 3:
        gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = ela_image
        
    std_dev = np.std(gray)
    score = 1.0 - (std_dev / 128.0)
    
    return float(np.clip(score, 0.0, 1.0))
