import os
import pandas as pd
from datetime import datetime
import streamlit as st
from preprocessing import decode_image_bytes, preprocess_image_bytes
import logging

from gradcam import get_backbone_submodel, make_gradcam_heatmap, overlay_heatmap

from exceptions import (
    PreprocessingError,
    ModelExecutionError,
)

from inference import (
    preprocess_image,
    preprocess_uploaded_image as _preprocess_uploaded_image,
    predict_image as _predict_image,
    find_last_conv_layer,
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

from utils.model_loader import load_cached_model , get_model_mtime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

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

.batch-summary-real {
    color: #22c55e;
    font-weight: 700;
}

.batch-summary-fake {
    color: #ef4444;
    font-weight: 700;
}

footer {
    visibility: hidden;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------- CONFIDENCE THRESHOLD ------------------------

LOW_CONFIDENCE_THRESHOLD = 0.70

# ----------------------- LOAD MODEL ------------------------

try:

    with st.spinner("Loading AI model..."):

        model = load_cached_model(get_model_mtime())

    st.success("Model initialized successfully.")

except Exception as e:

    logger.error(
        f"Model loading failed: {e}",
        exc_info=True
    )

    st.error(f"Error loading model: {str(e)}")

    model = None


# ----------------------- IMAGE PIPELINE --------------------

preprocess_uploaded_image = _preprocess_uploaded_image

try:

    preprocess_uploaded_image.cache_clear = preprocess_image_bytes.cache_clear
    preprocess_uploaded_image.cache_info = preprocess_image_bytes.cache_info

except Exception:
    pass

_ = preprocess_image


def predict_image(image):

    return _predict_image(model, image)


# ----------------------- HEADER / HERO ---------------------

st.markdown(
    "<h1 class='main-title'>DEEPFAKE SENTINEL</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>AI-powered detection of manipulated social media images.</p>",
    unsafe_allow_html=True,
)

if os.path.exists("coverpage.png"):

    st.image(
        "coverpage.png",
        use_container_width=True
    )

# ----------------------- TOP INFO SECTION ------------------

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

# ----------------------- TRAINING PERFORMANCE PLOTS --------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.subheader("📈 Training Performance")

col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    if os.path.exists("Figure_1.png"):
        st.image("Figure_1.png", use_container_width=True, caption="Training History")  # FIX #77: replaced deprecated use_column_width=True
    else:
        st.warning("Missing image: Figure_1.png")

with col_plot2:
    if os.path.exists("Figure_2.png"):
        st.image("Figure_2.png", use_container_width=True, caption="Evaluation Metrics")  # FIX #77: replaced deprecated use_column_width=True
    else:
        st.warning("Missing image: Figure_2.png")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- DETECTION SECTION -----------------

st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.3, 1])

with col_left:

    st.markdown("<div class='glass-card upload-box'>", unsafe_allow_html=True)

    st.subheader("🖼 Upload Images")

    uploaded_files = st.file_uploader(
        "Drop or browse social media images (select one or more)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.subheader("📊 Detection Results")

    if not uploaded_files:

        st.write(
            "Upload one or more images on the left to run deepfake detection."
        )

    elif model is None:

        st.error(
            "Model could not be loaded. Detection is unavailable."
        )

    else:

        # ---- constants ----
        MAX_FILE_SIZE_MB = 10
        MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

        # ---- per-batch accumulators ----
        batch_results = []
        batch_errors  = []

        # Initialise session history once
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        # ---- process every uploaded file ----
        progress_bar = st.progress(0, text="Analysing images…")

        for idx, uploaded_file in enumerate(uploaded_files):

            progress_bar.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Analysing {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})…"
            )

            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                batch_errors.append((
                    uploaded_file.name,
                    f"File too large ({uploaded_file.size / (1024 * 1024):.1f} MB). "
                    f"Maximum allowed is {MAX_FILE_SIZE_MB} MB."
                ))
                continue

            try:
                raw_bytes = uploaded_file.read()
                bgr_image = decode_image_bytes(raw_bytes)

            except Exception as e:
                batch_errors.append((uploaded_file.name, f"Could not read file: {e}"))
                continue

            label         = None
            confidence    = None
            processed_img = None

            try:
                processed_img = preprocess_uploaded_image(raw_bytes)
                prediction    = model.predict(processed_img, verbose=0)
                class_label   = int(np.argmax(prediction, axis=1)[0])
                confidence    = float(np.max(prediction))
                label         = "Real" if class_label == 0 else "Fake"

            except PreprocessingError as e:
                logger.error(f"PreprocessingError for {uploaded_file.name}: {e}", exc_info=True)
                batch_errors.append((uploaded_file.name, "Image preprocessing failed."))
                continue

            except ModelExecutionError as e:
                logger.error(f"ModelExecutionError for {uploaded_file.name}: {e}", exc_info=True)
                batch_errors.append((uploaded_file.name, "Model inference failed."))
                continue

            except Exception as e:
                logger.error(f"Unexpected error for {uploaded_file.name}: {e}", exc_info=True)
                batch_errors.append((uploaded_file.name, f"Unexpected error: {e}"))
                continue

            gradcam_image = None
            try:
                # Dynamic lookup avoids coupling Grad-CAM to layer ordering.
                backbone_model  = get_backbone_submodel(model)
                last_conv_layer = find_last_conv_layer(backbone_model)
                heatmap         = make_gradcam_heatmap(processed_img, backbone_model, last_conv_layer)
                gradcam_image   = overlay_heatmap(bgr_image, heatmap)
            except Exception as e:
                logger.warning(f"Grad-CAM failed for {uploaded_file.name}: {e}", exc_info=True)

            batch_results.append({
                "filename":     uploaded_file.name,
                "label":        label,
                "confidence":   confidence,
                "bgr_image":    bgr_image,
                "gradcam":      gradcam_image,
                "is_uncertain": confidence < LOW_CONFIDENCE_THRESHOLD,
            })

            st.session_state.prediction_history.append({
                "Filename":         uploaded_file.name,
                "Result":           label,
                "Confidence (%)":   f"{confidence * 100:.1f}",
                "Timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        if progress_bar is not None:
            progress_bar.empty()

        if batch_results:

            total     = len(batch_results)
            n_real    = sum(1 for r in batch_results if r["label"] == "Real")
            n_fake    = total - n_real
            avg_conf  = sum(r["confidence"] for r in batch_results) / total

            st.markdown("#### 📋 Batch Summary")

            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Total Analysed", total)
            s_col2.metric("✅ Real",  n_real)
            s_col3.metric("🚨 Fake",  n_fake)
            s_col4.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")

            st.markdown("---")

        for res in batch_results:

            is_uncertain = res["is_uncertain"]

            if is_uncertain:
                icon       = "🟡"
                verdict    = "Uncertain"
            elif res["label"] == "Real":
                icon       = "🟢"
                verdict    = "Authentic"
            else:
                icon       = "🔴"
                verdict    = "Deepfake suspected"

            expander_label = (
                f"{icon} {res['filename']} — {res['label']} "
                f"({res['confidence'] * 100:.1f}%)"
            )

            with st.expander(expander_label, expanded=(len(batch_results) == 1)):

                img_col, result_col = st.columns([1, 1])

                with img_col:
                    st.image(
                        res["bgr_image"],
                        channels="BGR",
                        caption="Uploaded image",
                        use_container_width=True,
                    )
                    if res["gradcam"] is not None:
                        st.image(
                            res["gradcam"],
                            channels="BGR",
                            caption="Grad-CAM attention map",
                            use_container_width=True,
                        )

                with result_col:
                    if is_uncertain:
                        style_class = "result-uncertain"
                        headline    = "Low Confidence — Uncertain"
                    elif res["label"] == "Real":
                        style_class = "result-real"
                        headline    = "Authentic image"
                    else:
                        style_class = "result-fake"
                        headline    = "Deepfake suspected"

                    st.markdown(
                        f"<div class='{style_class}' style='padding-left:0.8rem;'>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"### {icon} {headline}")
                    st.markdown(f"**Model prediction:** {res['label']}")
                    st.progress(res["confidence"])
                    st.caption(f"Confidence: {res['confidence'] * 100:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

        if batch_errors:

            st.markdown("---")
            st.warning(f"⚠️ {len(batch_errors)} file(s) could not be processed:")

            for fname, reason in batch_errors:
                st.error(f"**{fname}** — {reason}")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- PREDICTION HISTORY / CSV EXPORT --

if st.session_state.get("prediction_history"):

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🗂 Prediction History")

    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Report as CSV",
        data=csv_data,
        file_name=f"pixeltruth_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.markdown("</div>", unsafe_allow_html=True)
