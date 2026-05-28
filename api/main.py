import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our unified predict pipeline
from predict import predict_image
from exceptions import PreprocessingError, ModelExecutionError

logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


async def _read_image_bytes(file: UploadFile) -> bytes:
    chunks = []
    total_size = 0
    while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image is too large.")
        chunks.append(chunk)
    return b"".join(chunks)

app = FastAPI(
    title="PixelTruth API",
    description="Deepfake detection API that classifies an image as Real or Fake.",
    version="1.0.0",
)

# Allow CORS for external web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns deepfake detection results.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await _read_image_bytes(file)
        result = predict_image(image_bytes)
        return {
            "verdict": result["label"],
            "confidence": result["confidence"],
            "raw_scores": result["raw"]
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
