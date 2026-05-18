# 🔍 PixelTruth — AI-Powered Deepfake Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Accuracy-95%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/GSSoC-2026-purple?style=for-the-badge" />
</p>

> **PixelTruth** is an AI-powered deepfake detection system for social media images. It uses a custom Convolutional Neural Network (CNN) with advanced preprocessing to classify images as **real or AI-generated** with **95% accuracy**. Features interactive confidence visualization and comprehensive model analytics.

---

## 📌 Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [GSSoC 2026](#gssoc-2026)
- [License](#license)

---

## 🧠 About

With the rise of AI-generated media, detecting deepfakes has become critical for media integrity and combating misinformation. PixelTruth addresses this by providing a fast, accurate, and accessible deepfake detection tool built on deep learning.

It analyzes visual artifacts introduced during image synthesis and classifies images in real-time through an intuitive web dashboard with comprehensive confidence visualization and model explainability.

---

## ✨ Features

- 🖼️ **Real-time image analysis** — supports JPG, PNG, WebP formats
- 📊 **Confidence visualization** — interactive gauge, bar charts, and probability percentages
- 🎯 **Prediction transparency** — shows Real vs Fake probability distribution
- ⚠️ **Borderline detection** — alerts when prediction is close to decision boundary
- 📉 **Low confidence warnings** — alerts when model certainty is below 60%
- 🧠 **Grad-CAM explainability** — visual heatmap showing areas of focus
- 📈 **Model analytics dashboard** — confusion matrix, ROC curve, class distribution pie chart
- 🎨 **Modern Streamlit dashboard** — dark theme UI with visual feedback
- ⚡ **Fast inference** — lightweight model optimized for speed

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | CNN model training & inference |
| OpenCV | Image preprocessing |
| Streamlit | Web dashboard |
| NumPy / Matplotlib | Data handling & visualization |
| Seaborn | Statistical charts |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 95% |
| Precision | 94.8% |
| Recall | 95.7% |
| F1-Score | 95.2% |
| Input Size | 96 x 96 px |
| Supported Formats | JPG, PNG, WebP |

---

## ⚙️ Installation

> **⚠️ Prerequisites:**
> - **Python Version:** This project requires **Python 3.8+**.

```bash
# 1. Clone the repository
git clone https://github.com/Brijeshrath67/PixelTruth.git
cd PixelTruth

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Quick Start Script

```bash
# Just run the startup script
./run.sh
```

### Model Setup

PixelTruth needs a trained model file for predictions. A default model is included.

---

## 🚀 Usage

```bash
# Run the Streamlit dashboard
streamlit run app.py

# Or use the startup script
./run.sh
```

Then open your browser at `http://localhost:8501`

### How to Use

1. **Upload Image** — Drag and drop or browse for an image (JPG, PNG, or WebP)
2. **View Results** — See confidence gauge and probability bars
3. **Review Warnings** — Check for borderline or low confidence alerts
4. **Explore Grad-CAM** — View heatmap showing areas of focus
5. **Check Analytics** — Explore model performance metrics and charts

---

## 📊 Confidence Visualization

### Prediction Output Features

| Feature | Description |
|---------|-------------|
| **Confidence Gauge** | Semicircular gauge showing Real vs Fake probability |
| **Probability Bars** | Visual progress bars for each class |
| **Percentage Display** | Exact confidence values for Real and Fake |
| **Borderline Alerts** | Warning when prediction is uncertain (<15% margin) |
| **Low Confidence Alerts** | Info when model confidence < 60% |
| **Decision Explanation** | Human-readable explanation of predictions |
| **Grad-CAM Heatmap** | Visual attention map showing model focus areas |

---

## 📈 Model Analytics Dashboard

The app includes a comprehensive analytics section with:

- **Performance Metrics** — Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix** — Visual breakdown of predictions vs actual
- **ROC Curve** — Model discrimination ability visualization
- **Class Distribution** — Pie chart showing Real vs Fake ratio
- **Per-Class Statistics** — Detailed performance for each class

---

## 📁 Project Structure

```
PixelTruth/
├── app.py                 # Main Streamlit dashboard with confidence visualization
├── app_simple.py          # Simple test version of the app
├── test_app.py            # Testing version of the app
├── predict.py             # Inference logic
├── train.py               # Model training (v1)
├── train_v2.py            # Model training (v2)
├── train_v3.py            # Model training (v3)
├── train_quick.py         # Quick training script
├── gradcam.py             # Grad-CAM heatmap generation
├── metrics.py             # Analytics charts and metrics
├── model_utils.py         # Model file management utilities
├── requirements.txt       # Dependencies
├── Figure_1.png           # Training loss plot
├── Figure_2.png           # Training accuracy plot
├── deepfake_detection_model.h5  # Pre-trained model
├── run.sh                 # Startup script
└── README.md
```

---

## 🤝 Contributing

We welcome contributions of all kinds! Whether you're fixing a bug, improving the UI, or adding new features — you're welcome here.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🌸 GSSoC 2026

PixelTruth is participating in **GirlScript Summer of Code 2026**!

We have beginner-friendly issues ready. Look for issues labelled:
- `good-first-issue`
- `beginner-friendly`
- `documentation`
- `enhancement`

Feel free to explore open issues and start contributing!

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ for media integrity by <a href="https://github.com/Brijeshrath67">Brijesh Rathour</a>
  <br/>
  If you found this useful, please ⭐ star the repo!
</p>