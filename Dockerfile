# PixelTruth — one image for both the Streamlit UI and the FastAPI service.
#   docker build -t pixeltruth .
#   docker run -p 8501:8501 pixeltruth                       # Streamlit UI
#   docker run -p 8000:8000 pixeltruth \
#       uvicorn api.main:app --host 0.0.0.0 --port 8000      # REST API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIXELTRUTH_DB_PATH=/data/pixeltruth_history.db

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd --create-home pixeltruth && mkdir -p /data && chown pixeltruth /data
USER pixeltruth

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
