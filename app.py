import os
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from preprocessing import (
    decode_image_bytes,
    preprocess_image_bytes,
    detect_and_crop_face,
    preprocess_image_array,
)
import logging
import hashlib

from exif_analysis import extract_exif
from gradcam import get_backbone_submodel, make_gradcam_heatmap, overlay_heatmap
from ela_analysis import compute_ela, ela_uniformity_score

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
    load_cached_metrics,
    get_sample_metrics,
    get_confusion_matrix_plot,
    get_roc_curve_plot,
    get_dataset_distribution_plot,
    get_class_statistics,
    get_confusion_matrix_caption,
    get_roc_curve_caption,
    get_dataset_distribution_caption,
    get_evaluated_at,
    get_total_images,
)

from utils.model_loader import load_cached_model, get_model_mtime

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
MAX_HISTORY_ENTRIES = 500

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

# Initialise prediction history containers in session state (one-time)
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "prediction_history_hashes" not in st.session_state:
    st.session_state.prediction_history_hashes = set()
if "prediction_csv" not in st.session_state:
    st.session_state.prediction_csv = None


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
        use_column_width=True
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
        st.image("Figure_1.png", use_column_width=True, caption="Training History")
    else:
        st.warning("Missing image: Figure_1.png")

with col_plot2:
    if os.path.exists("Figure_2.png"):
        st.image("Figure_2.png", use_column_width=True, caption="Evaluation Metrics")
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
        MAX_FILE_SIZE_MB = 10
        MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

        batch_results = []
        batch_errors = []

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

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
                exif_data = extract_exif(raw_bytes)
                bgr_image = decode_image_bytes(raw_bytes)

            except Exception as e:
                batch_errors.append((uploaded_file.name, f"Could not read file: {e}"))
                continue

            label = None
            confidence = None
            processed_img = None
            face_image = None
            face_detected = False
            face_box = None
            box_image = bgr_image.copy()

            try:
                face_image, face_box = detect_and_crop_face(bgr_image)

                if face_box is not None:
                    face_detected = True
                    x, y, w, h = face_box

                    import cv2
                    cv2.rectangle(
                        box_image,
                        (x, y),
                        (x + w, y + h),
                        (94, 219, 120),
                        3
                    )

                processed_img = preprocess_image_array(face_image)
                prediction = model.predict(processed_img, verbose=0)

                from predict import decode_prediction
                label, confidence, _ = decode_prediction(prediction)

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
                # Dynamic lookup — avoids breaking Grad-CAM if model architecture changes
                backbone_model = get_backbone_submodel(model)
                last_conv_layer = find_last_conv_layer(backbone_model)
                heatmap = make_gradcam_heatmap(
                    processed_img,
                    backbone_model,
                    last_conv_layer
                )
                gradcam_image = overlay_heatmap(face_image, heatmap)

            except Exception as e:
                logger.warning(f"Grad-CAM failed for {uploaded_file.name}: {e}", exc_info=True)

            ela_image = None
            ela_score = None
            try:
                ela_image = compute_ela(raw_bytes)
                if ela_image is not None:
                    ela_score = ela_uniformity_score(ela_image)
            except Exception as e:
                logger.warning(f"ELA failed for {uploaded_file.name}: {e}")

            batch_results.append({
                "filename": uploaded_file.name,
                "label": label,
                "confidence": confidence,
                "bgr_image": bgr_image,
                "box_image": box_image,
                "face_image": face_image,
                "face_detected": face_detected,
                "gradcam": gradcam_image,
                "is_uncertain": confidence < LOW_CONFIDENCE_THRESHOLD,
                "exif": exif_data,
                "ela_image": ela_image,
                "ela_score": ela_score,
            })

            # Record history with deduplication and bounded size
            entry_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry_hash = hashlib.sha256(raw_bytes).hexdigest()
            if entry_hash not in st.session_state.prediction_history_hashes:
                history_entry = {
                    "Filename": uploaded_file.name,
                    "Result": label,
                    "Confidence (%)": f"{confidence * 100:.1f}",
                    "Timestamp": entry_timestamp,
                    "_hash": entry_hash,
                }
                st.session_state.prediction_history.append(history_entry)
                st.session_state.prediction_history_hashes.add(entry_hash)

                # Trim oldest entries when exceeding cap
                while len(st.session_state.prediction_history) > MAX_HISTORY_ENTRIES:
                    old = st.session_state.prediction_history.pop(0)
                    old_hash = old.get("_hash")
                    if old_hash and old_hash in st.session_state.prediction_history_hashes:
                        st.session_state.prediction_history_hashes.remove(old_hash)

            # Invalidate prepared CSV when history mutates
            st.session_state.prediction_csv = None

        progress_bar.empty()

        if batch_results:
            total = len(batch_results)
            n_real = sum(1 for r in batch_results if r["label"] == "Real")
            n_fake = total - n_real
            avg_conf = sum(r["confidence"] for r in batch_results) / total

            st.markdown("#### 📋 Batch Summary")

            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Total Analysed", total)
            s_col2.metric("✅ Real", n_real)
            s_col3.metric("🚨 Fake", n_fake)
            s_col4.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")

            st.markdown("---")

        for res in batch_results:
            is_uncertain = res["is_uncertain"]

            if is_uncertain:
                icon = "🟡"
            elif res["label"] == "Real":
                icon = "🟢"
            else:
                icon = "🔴"

            expander_label = (
                f"{icon} {res['filename']} — {res['label']} "
                f"({res['confidence'] * 100:.1f}%)"
            )

            with st.expander(expander_label, expanded=(len(batch_results) == 1)):
                img_col, result_col = st.columns([1.3, 1])

                with img_col:
                    if res["face_detected"]:
                        st.image(
                            res["box_image"],
                            channels="BGR",
                            caption="Uploaded image (face detected)",
                            use_column_width=True,
                        )

                        st.markdown(
                            "<div style='margin-top: 10px; margin-bottom: 5px; font-weight: 600;'>🔍 Model Input Analysis</div>",
                            unsafe_allow_html=True
                        )

                        crop_col1, crop_col2 = st.columns(2)

                        with crop_col1:
                            st.image(
                                res["face_image"],
                                channels="BGR",
                                caption="Detected face region",
                                use_column_width=True,
                            )

                        with crop_col2:
                            if res["gradcam"] is not None:
                                st.image(
                                    res["gradcam"],
                                    channels="BGR",
                                    caption="Grad-CAM face details",
                                    use_column_width=True,
                                )

                    else:
                        st.image(
                            res["bgr_image"],
                            channels="BGR",
                            caption="Uploaded image (no face detected, full image analyzed)",
                            use_column_width=True,
                        )

                        if res["gradcam"] is not None:
                            st.image(
                                res["gradcam"],
                                channels="BGR",
                                caption="Grad-CAM attention map (full image)",
                                use_column_width=True,
                            )

                    if res["ela_image"] is not None:
                        st.markdown(
                            "<div style='margin-top:10px; font-weight:600;'>"
                            "⚡ Error Level Analysis (ELA)"
                            "</div>",
                            unsafe_allow_html=True
                        )
                        ela_col1, ela_col2 = st.columns([1, 2])
                        with ela_col1:
                            st.image(res["ela_image"], channels="BGR",
                                     caption="ELA map", use_column_width=True)
                        with ela_col2:
                            score = res["ela_score"]
                            if score is not None:
                                if score > 0.75:
                                    ela_verdict = "🔴 High uniformity — AI pattern"
                                elif score > 0.5:
                                    ela_verdict = "🟡 Moderate uniformity — uncertain"
                                else:
                                    ela_verdict = "🟢 Non-uniform — natural photo pattern"
                                st.markdown(f"**ELA uniformity:** {ela_verdict}")
                                st.progress(score)
                                st.caption(
                                    f"Uniformity score: {score:.2f} (0 = natural, 1 = AI-like). "
                                    "AI-generated images often show uniform compression error "
                                    "across all regions."
                                )

                with result_col:
                    if is_uncertain:
                        style_class = "result-uncertain"
                        headline = "Low Confidence — Uncertain"
                    elif res["label"] == "Real":
                        style_class = "result-real"
                        headline = "Authentic image"
                    else:
                        style_class = "result-fake"
                        headline = "Deepfake suspected"

                    st.markdown(
                        f"<div class='{style_class}' style='padding-left:0.8rem;'>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(f"### {icon} {headline}")
                    st.markdown(f"**Model prediction:** {res['label']}")
                    st.progress(res["confidence"])
                    st.caption(f"Confidence: {res['confidence'] * 100:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("#### 🔍 Metadata Analysis")
                    exif = res["exif"]
                    
                    if exif["ai_software_detected"]:
                        exif_icon = "🔴"
                        label_text = f"AI software detected: {exif['software']}"
                    elif not exif["has_exif"]:
                        exif_icon = "🟡"
                        label_text = "No EXIF metadata"
                    else:
                        exif_icon = "🟢"
                        label_text = f"Camera: {exif.get('make','')} {exif.get('model','')}".strip()
                    
                    st.markdown(f"{exif_icon} **{label_text}**")
                    st.caption(exif["suspicion_reason"])
                    
                    if exif["has_exif"] and exif["field_count"]:
                        st.caption(f"{exif['field_count']} EXIF fields present"
                                   + (" · GPS data present" if exif["gps_present"] else ""))
                                   
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

    # Lightweight preview to avoid building a large DataFrame on every rerun
    preview = st.session_state.prediction_history[-50:]
    if preview:
        preview_df = pd.DataFrame(preview)
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.write("No recent history to preview.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("⬇️ Prepare CSV Report"):
            full_df = pd.DataFrame(st.session_state.prediction_history)
            st.session_state.prediction_csv = full_df.to_csv(index=False).encode("utf-8")
            st.success("Report prepared — click Download to save the CSV.")
    with c2:
        if st.button("🧹 Clear History"):
            st.session_state.prediction_history = []
            st.session_state.prediction_history_hashes = set()
            st.session_state.prediction_csv = None
            st.success("Prediction history cleared.")

    if st.session_state.get("prediction_csv") is not None:
        st.download_button(
            label="⬇️ Download Report as CSV",
            data=st.session_state.prediction_csv,
            file_name=f"pixeltruth_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- MODEL ANALYTICS ------------------

st.divider()
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📊 Model Analytics Dashboard")
st.caption("Comprehensive performance metrics and visualizations of the deepfake detection model")

cached_metrics = load_cached_metrics()

if cached_metrics is None:
    st.warning(
        "⚠️ No evaluation data found. The analytics dashboard requires "
        "real model evaluation results."
    )

    st.markdown(
        "Run the evaluation harness to generate real metrics:\n\n"
        "```bash\n"
        "python evaluate.py --test-dir path/to/test_data\n"
        "```\n\n"
        "The test directory should contain `real/` and `fake/` subdirectories "
        "with labelled images."
    )

else:
    evaluated_at = get_evaluated_at(cached_metrics)
    total_imgs = get_total_images(cached_metrics)

    if evaluated_at or total_imgs:
        meta_parts = []

        if evaluated_at:
            meta_parts.append(f"Evaluated at: {evaluated_at}")

        if total_imgs:
            meta_parts.append(f"{total_imgs:,} test images")

        st.caption(" | ".join(meta_parts))

    metrics = get_sample_metrics(cached_metrics)
    class_stats = get_class_statistics(cached_metrics)

    if metrics:
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

    st.markdown("#### 🎯 Classification Analysis")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm_fig = get_confusion_matrix_plot(cached_metrics)

        if cm_fig:
            st.plotly_chart(
                cm_fig,
                use_container_width=True,
                config={'scrollZoom': True, 'displayModeBar': True}
            )
            st.caption(get_confusion_matrix_caption())

    with col_roc:
        roc_fig = get_roc_curve_plot(cached_metrics)

        if roc_fig:
            st.plotly_chart(
                roc_fig,
                use_container_width=True,
                config={'scrollZoom': True, 'displayModeBar': True}
            )
            st.caption(get_roc_curve_caption())

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 📊 Data & Class-Level Insights")

    col_dist, col_stats = st.columns(2)

    with col_dist:
        dist_fig = get_dataset_distribution_plot(cached_metrics)

        if dist_fig:
            st.plotly_chart(
                dist_fig,
                use_container_width=True,
                config={'scrollZoom': True, 'displayModeBar': True}
            )
            st.caption(get_dataset_distribution_caption())

    with col_stats:
        if class_stats:
            st.markdown("**Per-Class Performance**")
            st.caption("Accuracy breakdown by image category")

            for idx, (class_label, stats) in enumerate(class_stats.items()):
                if idx > 0:
                    st.divider()

                icon = "🟢" if class_label == "Real" else "🔴"
                st.markdown(f"#### {icon} {class_label} Images")

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
    '''
<div style="text-align:center; margin-top:3rem; color:#6b7280; font-size:0.8rem;">
  <hr style="border-color:rgba(75,85,99,0.6);" />
  <p>🕵️ PixelTruth • Built with Streamlit & TensorFlow</p>
</div>
''',
    unsafe_allow_html=True,
)