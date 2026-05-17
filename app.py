import os
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from metrics import (
    get_sample_metrics,
    get_confusion_matrix_plot,
    get_roc_curve_plot,
    get_dataset_distribution_plot,
    get_class_statistics,
    get_confusion_matrix_caption,
    get_roc_curve_caption,
    get_dataset_distribution_caption,
)
from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256

st.set_page_config(
    page_title="PixelTruth",
    page_icon="🔍",
    layout="wide"
)

# ----------------------- CUSTOM CSS ------------------------
custom_css = """
<style>
.stApp {
    background: radial-gradient(circle at top left, #1d2671, #050816 40%, #000000 80%);
    color: #e5e7eb;
}
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#ff4b91,#facc15,#22c55e);
    -webkit-background-clip: text;
    color: transparent;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align:center;
    color:#9ca3af;
    font-size:0.95rem;
    margin-bottom: 1.8rem;
}
.glass-card {
    background: rgba(15,23,42,0.78);
    border-radius: 18px;
    padding: 1.3rem 1.6rem;
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 18px 45px rgba(15,23,42,0.9);
    backdrop-filter: blur(18px);
}
.result-real {
    border-left: 5px solid #22c55e;
}
.result-fake {
    border-left: 5px solid #ef4444;
}
.upload-box > div {
    border-radius: 18px !important;
    border: 1px dashed rgba(148,163,184,0.65) !important;
    background: rgba(15,23,42,0.6) !important;
}
.metric-small .stMetric {
    text-align: left;
}
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------- LOAD MODEL ------------------------
MODEL_PATH = get_model_path()
MODEL_URL = get_model_url()
MODEL_SHA256 = get_model_sha256()


@st.cache_resource
def load_deepfake_model():
    try:
        model_file_path = ensure_model_file(
            model_path=MODEL_PATH,
            model_url=MODEL_URL,
            model_sha256=MODEL_SHA256,
            download_if_missing=True,
        )
        return load_model(model_file_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_deepfake_model()


def render_missing_model_help():
    st.error(f"Model file '{MODEL_PATH}' not found in the current directory.")
    st.markdown(
        f"""
        ### Model setup required

        PixelTruth needs a trained Keras model before it can run predictions.

        1. Download or generate the model file.
        2. Save it as `{MODEL_PATH}` in the project root, or point `PIXELTRUTH_MODEL_PATH` to its location.
        3. Optionally set `PIXELTRUTH_MODEL_URL` to a GitHub Release asset or direct download link so the app can fetch it automatically on first run.
        4. Restart the app with `streamlit run app.py`.

        If you are using a release or shared model asset, place the downloaded file here:

        ```bash
        cp /path/to/downloaded/{MODEL_PATH} ./{MODEL_PATH}
        ```

        Optional checksum verification:

        ```bash
        export PIXELTRUTH_MODEL_SHA256=<sha256-from-release>
        ```
        """
    )

# ----------------------- IMAGE PIPELINE --------------------
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # fix: OpenCV loads BGR; model expects RGB (trained via PIL/ImageDataGenerator)
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict_image(image):
    if model is None:
        return None, None
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    class_label = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    label = "Real" if class_label == 0 else "Fake"
    return label, confidence

# ----------------------- HEADER / HERO ---------------------
st.markdown("<h1 class='main-title'>DEEPFAKE SENTINEL</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>AI‑powered detection of manipulated social media images.</p>",
    unsafe_allow_html=True,
)

if os.path.exists("coverpage.png"):
    st.image("coverpage.png", use_column_width=True)

# ----------------------- TOP INFO SECTION ------------------
col_info_left, col_info_right = st.columns([2, 1])

with col_info_left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧠 Understanding Deepfakes")
    st.markdown(
        """
- Deepfakes are AI‑generated images or videos where one person's face or identity is swapped with another.
- They can be used in entertainment and education, but also for misinformation, fraud, and privacy attacks.
- Detection models focus on subtle artifacts in lighting, edges, blending, and facial structure that humans often miss.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_info_right:
    st.markdown("<div class='glass-card metric-small'>", unsafe_allow_html=True)
    st.subheader("📈 Model Snapshot")
    st.metric("Training Accuracy", "95%")
    st.metric("Input Size", "96 × 96 pixels")
    st.metric("Task", "Binary classification (Real / Fake)")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- DETECTION SECTION -----------------
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.markdown("<div class='glass-card upload-box'>", unsafe_allow_html=True)
    st.subheader("🖼 Upload an Image")
    uploaded_file = st.file_uploader(
        "Drop or browse a social media image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(
                f"⚠️ File too large: **{uploaded_file.size / (1024 * 1024):.1f} MB**. "
                f"Please upload an image under {MAX_FILE_SIZE_MB} MB. "
                "Large RAW or TIFF files can crash the app — try a compressed JPG or PNG instead."
            )
            image = None
        else:
            try:
                raw_bytes = uploaded_file.read()
                file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
                uploaded_file.seek(0)  # reset file pointer after read
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                st.error(f"⚠️ Could not read the file: {e}. Please upload a valid JPG, PNG, or WebP image.")
                image = None

        if image is not None:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Preview")
            st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Detection Result")

    if uploaded_file is None:
        st.write("Upload an image on the left to run deepfake detection.")
    elif model is None:
        render_missing_model_help()
    else:
        with st.spinner("Analyzing image with the deepfake model..."):
            label, confidence = predict_image(image)

        if label is not None:
            style_class = "result-real" if label == "Real" else "result-fake"
            icon = "🟢" if label == "Real" else "🔴"
            headline = "Authentic image" if label == "Real" else "Deepfake suspected"

            st.markdown(f"<div class='{style_class}' style='padding-left:0.8rem;'>", unsafe_allow_html=True)
            st.markdown(f"### {icon} {headline}")
            st.markdown(f"**Model prediction:** {label}")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            if label == "Fake":
                st.error(
                    "The model detected patterns consistent with deepfake artifacts, "
                    "such as irregular blending, lighting mismatches, or unusual facial textures."
                )
            else:
                st.success(
                    "The model did not detect strong deepfake indicators. "
                    "The image appears consistent with natural, unaltered content."
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- MODEL PERFORMANCE -----------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("📉 Training Performance")

col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    st.markdown("**Training Accuracy Curve**")
    if os.path.exists("Figure_2.png"):
        st.image("Figure_2.png", use_container_width=True)
    else:
        st.info("Figure_2.png not found.")

with col_perf2:
    st.markdown("**Training Loss Curve**")
    if os.path.exists("Figure_1.png"):
        st.image("Figure_1.png", use_container_width=True)
    else:
        st.info("Figure_1.png not found.")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- MODEL ANALYTICS ------------------
st.divider()
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📊 Model Analytics Dashboard")
st.caption("Comprehensive performance metrics and visualizations of the deepfake detection model")

# Fetch metrics
metrics = get_sample_metrics()
class_stats = get_class_statistics()

# -------- SECTION 1: PERFORMANCE METRICS --------
st.markdown("#### 📈 Performance Metrics")
col_acc, col_prec, col_rec, col_f1 = st.columns(4)

with col_acc:
    st.metric(
        label="Accuracy",
        value=f"{metrics['accuracy']:.1f}%",
        help="Overall correctness: (TP + TN) / Total"
    )

with col_prec:
    st.metric(
        label="Precision",
        value=f"{metrics['precision']:.1f}%",
        help="Positive accuracy: TP / (TP + FP)"
    )

with col_rec:
    st.metric(
        label="Recall",
        value=f"{metrics['recall']:.1f}%",
        help="True positive rate: TP / (TP + FN)"
    )

with col_f1:
    st.metric(
        label="F1-Score",
        value=f"{metrics['f1_score']:.1f}%",
        help="Harmonic mean of precision & recall"
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# -------- SECTION 2: CONFUSION MATRIX & ROC CURVE --------
st.markdown("#### 🎯 Classification Analysis")
col_cm, col_roc = st.columns(2)

with col_cm:
    st.pyplot(get_confusion_matrix_plot(), use_container_width=True)
    st.caption(get_confusion_matrix_caption())

with col_roc:
    st.pyplot(get_roc_curve_plot(), use_container_width=True)
    st.caption(get_roc_curve_caption())

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# -------- SECTION 3: DATASET DISTRIBUTION & CLASS STATS --------
st.markdown("#### 📊 Data & Class-Level Insights")
col_dist, col_stats = st.columns(2)

with col_dist:
    st.pyplot(get_dataset_distribution_plot(), use_container_width=True)
    st.caption(get_dataset_distribution_caption())

with col_stats:
    st.markdown("**Per-Class Performance**")
    st.caption("Accuracy breakdown by image category")
    
    for idx, (class_label, stats) in enumerate(class_stats.items()):
        if idx > 0:
            st.divider()
        
        # Class header with icon
        icon = "🟢" if class_label == "Real" else "🔴"
        st.markdown(f"#### {icon} {class_label} Images")
        
        # Metrics in 3 columns
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric(
                label="Total Samples",
                value=f"{stats['total_samples']:,}"
            )
        
        with col_s2:
            st.metric(
                label="Correct Predictions",
                value=f"{stats['correctly_classified']:,}"
            )
        
        with col_s3:
            st.metric(
                label="Accuracy",
                value=f"{stats['class_accuracy']:.1f}%"
            )

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- FOOTER ----------------------------
st.markdown(
    """
<div style="text-align:center; margin-top:3rem; color:#6b7280; font-size:0.8rem;">
  <hr style="border-color:rgba(75,85,99,0.6);" />
  <p>🕵️ PixelTruth • Built with Streamlit & TensorFlow</p>
 
</div>
""",
    unsafe_allow_html=True,
)