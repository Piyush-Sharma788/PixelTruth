import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import logging

from preprocessing import (
    decode_image_bytes,
    preprocess_image_array,
    preprocess_image_bytes,
)

from utils.model_loader import load_cached_model

from gradcam import (
    make_gradcam_heatmap,
    overlay_heatmap,
    find_last_conv_layer,
)

from exceptions import (
    PreprocessingError,
    ModelExecutionError,
)

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PixelTruth",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

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
.result-uncertain {
    border-left: 5px solid #f59e0b;
}
.upload-box > div {
    border-radius: 18px !important;
    border: 1px dashed rgba(148,163,184,0.65) !important;
    background: rgba(15,23,42,0.6) !important;
}
.metric-small .stMetric {
    text-align: left;
}
footer {
    visibility: hidden;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ---------------- CONFIDENCE THRESHOLD ----------------

LOW_CONFIDENCE_THRESHOLD = 0.70

# ---------------- LOAD MODEL ----------------

with st.spinner("Loading AI model and warming up..."):

    try:

        model = load_cached_model()

        st.toast("✅ Model warmed up successfully!")

    except Exception as e:

        st.error(f"Error loading model: {str(e)}")

        model = None

# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_image(image):

    return preprocess_image_array(image)


def preprocess_uploaded_image(image_bytes):

    return preprocess_image_bytes(image_bytes)


preprocess_uploaded_image.cache_clear = preprocess_image_bytes.cache_clear
preprocess_uploaded_image.cache_info = preprocess_image_bytes.cache_info

# ---------------- PREDICTION ----------------

def predict_image(image):

    if model is None:
        return None, None, None

    processed_image = preprocess_image(image)

    try:

        prediction = model.predict(processed_image, verbose=0)

        class_label = np.argmax(prediction, axis=1)[0]

        confidence = float(np.max(prediction))

        label = "Real" if class_label == 0 else "Fake"

        return label, confidence, processed_image

    except Exception as e:

        logger.error(f"Model inference failed: {e}", exc_info=True)

        raise ModelExecutionError(
            f"Model prediction failed: {str(e)}"
        ) from e

# ---------------- HEADER ----------------

st.markdown(
    "<h1 class='main-title'>DEEPFAKE SENTINEL</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>AI-powered detection of manipulated social media images.</p>",
    unsafe_allow_html=True,
)

if os.path.exists("coverpage.png"):

    st.image("coverpage.png", use_column_width=True)

# ---------------- TOP INFO SECTION ----------------

col_info_left, col_info_right = st.columns([2, 1])

with col_info_left:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.subheader("🧠 Understanding Deepfakes")

    st.markdown(
        """
- Deepfakes are AI-generated images or videos where one person's face or identity is swapped with another.
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

# ---------------- DETECTION SECTION ----------------

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

    uploaded_image_bytes = None

    if uploaded_file is not None:

        if uploaded_file.size > MAX_FILE_SIZE_BYTES:

            st.error(
                f"⚠️ File too large: "
                f"{uploaded_file.size / (1024 * 1024):.1f} MB"
            )

            image = None

        else:

            try:

                raw_bytes = uploaded_file.read()

                uploaded_image_bytes = raw_bytes

                uploaded_file.seek(0)

                image = decode_image_bytes(raw_bytes)

            except Exception as e:

                st.error(
                    f"⚠️ Could not read the file: {e}"
                )

                image = None

        if image is not None:

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

            st.subheader("🔍 Preview")

            st.image(
                image,
                channels="BGR",
                caption="Uploaded Image",
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

with col_right:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.subheader("📊 Detection Result")

    if uploaded_file is None:

        st.write("Upload an image to run deepfake detection.")

    elif model is None:

        st.error("Model could not be loaded.")

    elif image is None:

        st.info("Upload a valid image.")

    else:

        with st.spinner("Analyzing image with the deepfake model..."):

            try:

                if uploaded_image_bytes is not None:

                    processed_image = preprocess_uploaded_image(
                        uploaded_image_bytes
                    )

                    prediction = model.predict(
                        processed_image,
                        verbose=0
                    )

                    class_label = np.argmax(prediction, axis=1)[0]

                    confidence = float(np.max(prediction))

                    label = "Real" if class_label == 0 else "Fake"

                else:

                    label, confidence, processed_image = predict_image(image)

            except PreprocessingError as e:

                logger.error(
                    f"Caught PreprocessingError in UI: {e}",
                    exc_info=True
                )

                st.error(
                    "⚠️ There was an issue processing the uploaded image."
                )

                label, confidence, processed_image = None, None, None

            except ModelExecutionError as e:

                logger.error(
                    f"Caught ModelExecutionError in UI: {e}",
                    exc_info=True
                )

                st.error(
                    "⚠️ The AI model encountered an error during analysis."
                )

                label, confidence, processed_image = None, None, None

            except Exception as e:

                logger.error(
                    f"Caught unexpected Runtime error in UI: {e}",
                    exc_info=True
                )

                st.error(
                    "⚠️ An unexpected runtime error occurred."
                )

                label, confidence, processed_image = None, None, None

        if label is not None:

            if "prediction_history" not in st.session_state:

                st.session_state.prediction_history = []

            st.session_state.prediction_history.append({
                "Filename": uploaded_file.name,
                "Result": label,
                "Confidence (%)": f"{confidence * 100:.1f}",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # ---------------- GRAD-CAM ----------------

            try:

                backbone_model = model.layers[0]

                last_conv_layer = find_last_conv_layer(
                    backbone_model
                )

                heatmap = make_gradcam_heatmap(
                    processed_image,
                    backbone_model,
                    last_conv_layer
                )

                gradcam_image = overlay_heatmap(
                    image,
                    heatmap
                )

            except Exception as e:

                logger.warning(
                    f"Grad-CAM visualization failed: {e}",
                    exc_info=True
                )

                gradcam_image = None

            # ---------------- RESULT STYLING ----------------

            is_uncertain = confidence < LOW_CONFIDENCE_THRESHOLD

            if is_uncertain:

                style_class = "result-uncertain"
                icon = "🟡"
                headline = "Low Confidence — Uncertain"

            elif label == "Real":

                style_class = "result-real"
                icon = "🟢"
                headline = "Authentic image"

            else:

                style_class = "result-fake"
                icon = "🔴"
                headline = "Deepfake suspected"

            st.markdown(
                f"<div class='{style_class}' style='padding-left:0.8rem;'>",
                unsafe_allow_html=True
            )

            st.markdown(f"### {icon} {headline}")

            st.markdown(f"**Model prediction:** {label}")

            st.progress(confidence)

            st.caption(f"Confidence: {confidence * 100:.1f}%")

            st.markdown("</div>", unsafe_allow_html=True)

            # ---------------- EXPLANATION ----------------

            if is_uncertain:

                st.warning(
                    "Prediction confidence is low. "
                    "Try another image for better accuracy."
                )

            elif label == "Fake":

                st.error(
                    "The model detected patterns consistent with deepfake artifacts."
                )

            else:

                st.success(
                    "The image appears consistent with natural content."
                )

            # ---------------- GRAD-CAM DISPLAY ----------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("🧠 Model Explainability (Grad-CAM)")

            col_gc1, col_gc2 = st.columns(2)

            with col_gc1:

                st.image(
                    image,
                    channels="BGR",
                    caption="Original Image",
                    use_column_width=True
                )

            with col_gc2:

                if gradcam_image is not None:

                    st.image(
                        gradcam_image,
                        channels="BGR",
                        caption="Grad-CAM Heatmap",
                        use_column_width=True
                    )

                else:

                    st.info(
                        "Grad-CAM visualization unavailable."
                    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MODEL ANALYTICS ----------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.markdown("### 📊 Model Analytics Dashboard")

metrics = get_sample_metrics()

class_stats = get_class_statistics()

col_acc, col_prec, col_rec, col_f1 = st.columns(4)

with col_acc:
    st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")

with col_prec:
    st.metric("Precision", f"{metrics['precision']:.1f}%")

with col_rec:
    st.metric("Recall", f"{metrics['recall']:.1f}%")

with col_f1:
    st.metric("F1 Score", f"{metrics['f1_score']:.1f}%")

st.divider()

col_cm, col_roc = st.columns(2)

with col_cm:

    st.plotly_chart(
        get_confusion_matrix_plot(),
        use_container_width=True
    )

    st.caption(get_confusion_matrix_caption())

with col_roc:

    st.plotly_chart(
        get_roc_curve_plot(),
        use_container_width=True
    )

    st.caption(get_roc_curve_caption())

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION HISTORY ----------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.subheader("🕒 Prediction History")

if (
    "prediction_history" not in st.session_state
    or len(st.session_state.prediction_history) == 0
):

    st.info("No predictions yet.")

else:

    df = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Report as CSV",
        data=csv,
        file_name="pixeltruth_report.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear History"):

        st.session_state.prediction_history = []

        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown(
    """
<div style="text-align:center; margin-top:3rem; color:#6b7280; font-size:0.8rem;">
  <hr style="border-color:rgba(75,85,99,0.6);" />
  <p>🕵️ PixelTruth • Built with Streamlit & TensorFlow</p>
</div>
""",
    unsafe_allow_html=True,
)