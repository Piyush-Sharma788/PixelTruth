#!/bin/bash
echo "Starting PixelTruth..."
cd "$(dirname "$0")"
streamlit run app.py --server.headless true --server.port 8501