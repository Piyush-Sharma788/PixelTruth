import os
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image  # CHANGED: added PIL — required for cv2→PIL conversion before HF pipeline
from predict import predict_image  # CHANGED: import HF-backed predict_image from predict.py

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

# CHANGED: replaced H5 loader + TF preprocessing pipeline with HF pipeline backed by predict.py
# CHANGED: removed load_deepfake_model(), get_model_input_size(), preprocess_image(), and old predict_image()

@st.cache_resource
def _load_hf_pipeline():
    # CHANGED: warm up the HF pipeline once at startup so the first upload isn't slow
    from predict import _pipe
    return _pipe

_load_hf_pipeline()  # CHANGED: triggers model auto-download/cache on app start

def _run_prediction(bgr_image):
    # CHANGED: convert cv2 BGR ndarray to PIL RGB expected by the HF pipeline
    pil_image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    return predict_image(pil_image)
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
    st.metric("Model Input", "224 × 224 pixels")  # CHANGED: HF model uses 224x224; removed get_model_input_size() call
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

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Preview")
            st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Could not read the uploaded image. Please try another file.")

with col_right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Detection Result")

    if uploaded_file is None:
        st.write("Upload an image on the left to run deepfake detection.")
    else:
        with st.spinner("Analyzing image with the deepfake model..."):
            label, confidence = _run_prediction(image)  # CHANGED: call _run_prediction (cv2→PIL→HF) instead of old predict_image

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
        st.image("Figure_2.png", use_column_width=True)
    else:
        st.info("Figure_2.png not found.")

with col_perf2:
    st.markdown("**Training Loss Curve**")
    if os.path.exists("Figure_1.png"):
        st.image("Figure_1.png", use_column_width=True)
    else:
        st.info("Figure_1.png not found.")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- FOOTER ----------------------------
st.markdown(
    """
<div style="text-align:center; margin-top:3rem; color:#6b7280; font-size:0.8rem;">
  <hr style="border-color:rgba(75,85,99,0.6);" />
  <p>🕵️ PixelTruth • Built with Streamlit &amp; Hugging Face Transformers</p>  <!-- CHANGED: updated footer — TensorFlow replaced by Hugging Face Transformers -->
 
</div>
""",
    unsafe_allow_html=True,
)

