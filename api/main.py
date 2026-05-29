import logging
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our unified predict pipeline
from predict import predict_image
from exceptions import PreprocessingError, ModelExecutionError

logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024

# ---------------------------------------------------------------------------
# CORS — explicit origin allowlist instead of wildcard
# ---------------------------------------------------------------------------
_DEFAULT_ORIGINS = "http://localhost:8501,http://localhost:8000"
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("PIXELTRUTH_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Optional API key authentication
# ---------------------------------------------------------------------------
API_KEY: str = os.getenv("PIXELTRUTH_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Rate limiting — 10 requests per minute per client IP
# ---------------------------------------------------------------------------
RATE_LIMIT: str = os.getenv("PIXELTRUTH_RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)


async def _read_image_bytes(file: UploadFile) -> bytes:
    chunks = []
    total_size = 0
    while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    retry_after = exc.detail.split()[-1] if exc.detail else "60"
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
        headers={"Retry-After": retry_after},
    )


app = FastAPI(
    title="PixelTruth API",
    description="Deepfake detection API that classifies an image as Real or Fake.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def _verify_api_key(request: Request) -> None:
    if not API_KEY:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set the X-API-Key header.",
        )


@app.post("/api/detect")
@limiter.limit(RATE_LIMIT)
async def detect_image(request: Request, file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns deepfake detection results.
    """
    _verify_api_key(request)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await _read_image_bytes(file)
        result = predict_image(image_bytes)
        return {
            "verdict": result["label"],
            "confidence": result["confidence"],
            "raw_scores": result["raw"],
            "face_detected": result.get("face_detected", False),
            "face_box": result.get("face_box", None)
        }

    except HTTPException:
        raise
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except ModelExecutionError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model execution.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
