import logging
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our unified predict pipeline
from predict import predict_image
from exceptions import PreprocessingError, ModelExecutionError

logger = logging.getLogger(__name__)

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
        # Read raw image bytes
        image_bytes = await file.read()
        
        # Get extension to ensure supported format
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
            
        try:
            # Pass the temp file path instead of bytes
            result = predict_image(tmp_path)
            
            return {
                "verdict": result["label"],
                "confidence": result["confidence"],
                "raw_scores": result["raw"]
            }
        finally:
            # Ensure the temp file is cleaned up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except ModelExecutionError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model execution.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
