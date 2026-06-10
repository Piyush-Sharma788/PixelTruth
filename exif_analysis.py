"""EXIF metadata extraction for deepfake secondary signal detection."""
from __future__ import annotations
from io import BytesIO
from PIL import Image

KNOWN_AI_SOFTWARE_TAGS = [
    "dall-e", "stable diffusion", "midjourney", "adobe firefly",
    "generative", "ai-generated", "imagen", "runway", "sora",
]

def extract_exif(image_bytes: bytes) -> dict:
    """
    Returns:
      {
        "has_exif": bool,
        "field_count": int,           # number of EXIF tags present
        "software": str | None,       # EXIF tag 305 (Software)
        "make": str | None,           # EXIF tag 271 (camera Make)
        "model": str | None,          # EXIF tag 272 (camera Model)
        "datetime": str | None,       # EXIF tag 306
        "gps_present": bool,          # any GPS tags (34853)
        "ai_software_detected": bool, # True if software tag matches known AI
        "suspicious": bool,           # True if no EXIF at all OR AI software found
        "suspicion_reason": str,      # human-readable reason for UI display
      }
    """
    result = {
        "has_exif": False,
        "field_count": 0,
        "software": None,
        "make": None,
        "model": None,
        "datetime": None,
        "gps_present": False,
        "ai_software_detected": False,
        "suspicious": True,
        "suspicion_reason": "No EXIF metadata found \u2014 consistent with AI generation"
    }
    
    try:
        img = Image.open(BytesIO(image_bytes))
        exif = img._getexif()
    except Exception:
        exif = None
        
    if exif is None:
        return result
        
    result["has_exif"] = True
    result["field_count"] = len(exif)
    result["suspicious"] = False
    result["suspicion_reason"] = "Camera metadata present \u2014 consistent with a real photograph"
    
    # 305: Software, 271: Make, 272: Model, 306: DateTime, 34853: GPSInfo
    result["software"] = exif.get(305)
    result["make"] = exif.get(271)
    result["model"] = exif.get(272)
    result["datetime"] = exif.get(306)
    result["gps_present"] = 34853 in exif
    
    if result["software"]:
        software_lower = str(result["software"]).lower()
        for ai_tag in KNOWN_AI_SOFTWARE_TAGS:
            if ai_tag in software_lower:
                result["ai_software_detected"] = True
                result["suspicious"] = True
                result["suspicion_reason"] = f"Software tag identifies an AI image generator: {result['software']}"
                break
                
    return result

def format_exif_summary(exif: dict) -> str:
    """One-line human-readable summary for UI display."""
    if not exif["has_exif"]:
        return "No EXIF metadata found."
    if exif["ai_software_detected"]:
        return f"AI generation software detected: {exif['software']}."
    summary = f"Camera metadata found ({exif['field_count']} fields)."
    if exif["make"] or exif["model"]:
        summary += f" Camera: {exif.get('make', '')} {exif.get('model', '')}".strip() + "."
    if exif["gps_present"]:
        summary += " Includes GPS data."
    return summary
