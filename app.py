import os
from dotenv import load_dotenv
load_dotenv()
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import streamlit as st
from preprocessing import decode_image_bytes
import logging
import hashlib

from history import init_db, save_prediction, load_history, clear_history
from exif_analysis import extract_exif
from gradcam import get_backbone_submodel, make_gradcam_heatmap, overlay_heatmap, find_last_conv_layer
from ela_analysis import compute_ela, ela_uniformity_score
from calibration import temperature_scale
from forensic_filters import compute_luminance_gradient, compute_noise_residual

from exceptions import (
    PreprocessingError,
    ModelExecutionError,
)

from predict import predict_image as _shared_predict_image, decode_prediction

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
from config import (
    DEFAULT_TEMPERATURE,
    LOG_FORMAT,
    LOW_CONFIDENCE_THRESHOLD as _DEFAULT_CONFIDENCE_THRESHOLD,
    MODEL_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)
def sanitize_csv_value(value):
    """Prevent CSV formula injection by prefixing dangerous characters."""
    if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')):
        return "'" + value
    return value


def utc_timestamp() -> str:
    """UTC ISO-8601 timestamp, same format history.save_prediction stores."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Persist history to SQLite (default on). For a public, multi-user deployment
# set PIXELTRUTH_PERSIST_HISTORY=0 so each browser session only sees its own
# predictions -- the SQLite table is shared by every visitor otherwise.
PERSIST_HISTORY = os.getenv("PIXELTRUTH_PERSIST_HISTORY", "1").strip().lower() not in ("0", "false", "no")

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

# ----------------------- CONFIGURATION SIDEBAR -----------------------

st.sidebar.header("⚙️ Configuration")
LOW_CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=1.00,
    value=_DEFAULT_CONFIDENCE_THRESHOLD,
    step=0.05,
    help="Predictions with confidence below this threshold will be flagged as uncertain."
)
CALIBRATION_TEMPERATURE = st.sidebar.slider(
    "Calibration Temperature (T)",
    min_value=0.5,
    max_value=3.0,
    value=float(DEFAULT_TEMPERATURE),
    step=0.05,
    help=(
        "Rescales confidence without changing the Real/Fake verdict. "
        "T > 1 softens, T < 1 sharpens. The default is the value fitted on the "
        "validation set (see deepfake_detection_model.json)."
    ),
)

MAX_HISTORY_ENTRIES = 500

# ----------------------- DATABASE INIT ----------------------

try:
    if PERSIST_HISTORY:
        init_db()
except Exception as e:
    logger.warning(f"Could not initialize prediction history database: {e}", exc_info=True)

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

    st.error(
        f"Error loading model: {e}\n\n"
        "Make sure `deepfake_detection_model.h5` is in the project root "
        "(it ships with the repository), or set `PIXELTRUTH_MODEL_PATH` / "
        "`PIXELTRUTH_MODEL_URL`. See README → Model setup."
    )

    model = None


# ----------------------- IMAGE PIPELINE --------------------

# Initialise prediction history containers in session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "prediction_history_hashes" not in st.session_state:
    st.session_state.prediction_history_hashes = set()

if "prediction_csv" not in st.session_state:
    st.session_state.prediction_csv = None

# Load persisted history from DB once
if "history_loaded_from_db" not in st.session_state:
    try:
        persisted_rows = load_history(limit=MAX_HISTORY_ENTRIES) if PERSIST_HISTORY else []

        if persisted_rows and not st.session_state.prediction_history:
            # load_history returns newest first; session history is kept
            # oldest -> newest so new predictions can be appended.
            st.session_state.prediction_history = list(reversed(persisted_rows))

        # Populate prediction_history_hashes with the hashes of loaded history
        for entry in st.session_state.prediction_history:
            h = entry.get("_hash")
            if h:
                st.session_state.prediction_history_hashes.add(h)

        st.session_state.history_loaded_from_db = True

    except Exception as e:
        logger.warning(f"Could not load prediction history: {e}", exc_info=True)
        st.session_state.history_loaded_from_db = True


def predict_image(image, **kwargs):
    """Thin wrapper around the shared predict_image; forwards all kwargs (e.g. temperature)."""
    return _shared_predict_image(image, **kwargs)


# ----------------------- HEADER / HERO ---------------------

st.markdown(
    "<h1 class='main-title'>PIXELTRUTH</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>AI-powered detection of manipulated social media images.</p>",
    unsafe_allow_html=True,
)

if os.path.exists("coverpage.png"):
    st.image(
        "coverpage.png",
        width="stretch"
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

    _snapshot_metrics = load_cached_metrics()
    if _snapshot_metrics:
        st.metric(
            "Test Accuracy",
            f"{_snapshot_metrics['accuracy']:.1f}%",
            help=f"Measured by evaluate.py on {_snapshot_metrics.get('total_images', '?'):,} held-out test images.",
        )
    else:
        st.metric("Test Accuracy", "Not evaluated", help="Run evaluate.py to measure it.")
    st.metric("Input Size", "96 × 96 pixels (face crop)")
    st.metric("Task", "Binary classification (Real / Fake)")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- TRAINING PERFORMANCE PLOTS --------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.subheader("📈 Training Performance")

col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    if os.path.exists("Figure_1.png"):
        st.image("Figure_1.png", width="stretch", caption="Training history: loss")
    else:
        st.warning("Missing image: Figure_1.png")

with col_plot2:
    if os.path.exists("Figure_2.png"):
        st.image("Figure_2.png", width="stretch", caption="Training history: accuracy")
    else:
        st.warning("Missing image: Figure_2.png")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- DETECTION SECTION -----------------

st.markdown("<br>", unsafe_allow_html=True)



analysis_mode = st.radio(
    "🔍 Analysis Mode",
    ["Batch Analysis", "Forensic Comparison"],
    horizontal=True,
)

st.markdown("<br>", unsafe_allow_html=True)

if analysis_mode == "Batch Analysis":

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

        if uploaded_files is None:
            uploaded_files = []
        else:
            uploaded_files = [f for f in uploaded_files if f is not None]

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

            if "current_predictions" not in st.session_state:
                st.session_state.current_predictions = {}

            batch_results = []
            batch_errors = []

            # One entry per upload, keyed by content hash (not filename), so two
            # different files that share a name never overwrite each other.
            upload_entries = []  # (uploaded_file, raw_bytes, sha256)
            for uploaded_file in uploaded_files:
                if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                    batch_errors.append((
                        uploaded_file.name,
                        f"File too large ({uploaded_file.size / (1024 * 1024):.1f} MB). "
                        f"Maximum allowed is {MAX_FILE_SIZE_MB} MB."
                    ))
                    continue

                try:
                    raw_bytes = uploaded_file.getvalue()
                    upload_entries.append(
                        (uploaded_file, raw_bytes, hashlib.sha256(raw_bytes).hexdigest())
                    )
                except Exception as e:
                    batch_errors.append((uploaded_file.name, f"Could not read file: {e}"))

            uploaded_hashes = {h for _, _, h in upload_entries}

            # Drop cached results for files that are no longer uploaded
            st.session_state.current_predictions = {
                h: res for h, res in st.session_state.current_predictions.items() if h in uploaded_hashes
            }

            hashes_to_process = []
            for _, _, h in upload_entries:
                if h not in st.session_state.current_predictions and h not in hashes_to_process:
                    hashes_to_process.append(h)

            progress_bar = None
            if hashes_to_process:
                progress_bar = st.progress(0, text="Analysing images…")

            for uploaded_file, raw_bytes, entry_hash in upload_entries:
                # Already analysed: only re-apply the (possibly changed) temperature.
                if entry_hash in st.session_state.current_predictions:
                    cached_res = dict(st.session_state.current_predictions[entry_hash])
                    calibrated_pred = temperature_scale(cached_res["raw_prediction"], temperature=CALIBRATION_TEMPERATURE)
                    label, confidence, _ = decode_prediction(calibrated_pred)
                    cached_res["filename"] = uploaded_file.name
                    cached_res["label"] = label
                    cached_res["confidence"] = confidence
                    cached_res["is_uncertain"] = confidence < LOW_CONFIDENCE_THRESHOLD
                    batch_results.append(cached_res)
                    continue

                if progress_bar is not None:
                    process_idx = hashes_to_process.index(entry_hash)
                    progress_bar.progress(
                        (process_idx + 1) / len(hashes_to_process),
                        text=f"Analysing {uploaded_file.name} ({process_idx + 1}/{len(hashes_to_process)})…"
                    )

                try:
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
                    # Run prediction — face detection happens once inside predict_image.
                    # We reuse its face_image / face_box rather than calling
                    # detect_and_crop_face a second time.
                    prediction = predict_image(raw_bytes, temperature=CALIBRATION_TEMPERATURE)
                    label = prediction["label"]
                    confidence = prediction["confidence"]
                    processed_img = prediction["processed_image"]
                    raw_pred_array = prediction["raw_prediction"]
                    face_image = prediction.get("face_image", bgr_image)
                    face_box = prediction.get("face_box")

                    if face_box is not None:
                        face_detected = True
                        x, y, w, h = face_box
                        cv2.rectangle(
                            box_image,
                            (x, y),
                            (x + w, y + h),
                            (94, 219, 120),
                            3
                        )

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
                    backbone_model = get_backbone_submodel(model)
                    last_conv_layer = find_last_conv_layer(backbone_model)

                    heatmap = make_gradcam_heatmap(
                        processed_img,
                        model,
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

                gradient_image = None
                try:
                    gradient_image = compute_luminance_gradient(raw_bytes)
                except Exception as e:
                    logger.warning(f"Luminance gradient failed for {uploaded_file.name}: {e}")

                noise_residual_image = None
                try:
                    noise_residual_image = compute_noise_residual(raw_bytes)
                except Exception as e:
                    logger.warning(f"Noise residual analysis failed for {uploaded_file.name}: {e}")

                prediction_result = {
                    "filename": uploaded_file.name,
                    "label": label,
                    "confidence": confidence,
                    "raw_prediction": raw_pred_array,
                    "bgr_image": bgr_image,
                    "box_image": box_image,
                    "face_image": face_image,
                    "face_detected": face_detected,
                    "gradcam": gradcam_image,
                    "is_uncertain": confidence < LOW_CONFIDENCE_THRESHOLD,
                    "exif": exif_data,
                    "ela_image": ela_image,
                    "ela_score": ela_score,
                    "gradient_image": gradient_image,
                    "noise_residual_image": noise_residual_image,
                }

                batch_results.append(prediction_result)
                st.session_state.current_predictions[entry_hash] = prediction_result

                if entry_hash not in st.session_state.prediction_history_hashes:
                    # Same columns/format as history.load_history so session
                    # and persisted rows look identical. CSV escaping is
                    # applied at export time, not here.
                    history_entry = {
                        "Filename": uploaded_file.name,
                        "Result": label,
                        "Confidence (%)": f"{confidence * 100:.1f}",
                        "Timestamp": utc_timestamp(),
                        "Face Detected": bool(face_detected),
                        "_hash": entry_hash,
                    }

                    st.session_state.prediction_history.append(history_entry)
                    st.session_state.prediction_history_hashes.add(entry_hash)

                    if PERSIST_HISTORY:
                        try:
                            save_prediction(
                                filename=uploaded_file.name,
                                verdict=label,
                                confidence_pct=round(confidence * 100, 1),
                                face_detected=int(face_detected),
                                image_hash=entry_hash,
                            )
                        except Exception as e:
                            logger.warning(f"Could not save prediction history: {e}", exc_info=True)

                    while len(st.session_state.prediction_history) > MAX_HISTORY_ENTRIES:
                        old = st.session_state.prediction_history.pop(0)
                        old_hash = old.get("_hash")

                        if old_hash and old_hash in st.session_state.prediction_history_hashes:
                            st.session_state.prediction_history_hashes.remove(old_hash)

                st.session_state.prediction_csv = None

            if progress_bar is not None:
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
                                width="stretch",
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
                                    width="stretch",
                                )

                            with crop_col2:
                                if res["gradcam"] is not None:
                                    st.image(
                                        res["gradcam"],
                                        channels="BGR",
                                        caption="Grad-CAM face details",
                                        width="stretch",
                                    )

                        else:
                            st.image(
                                res["bgr_image"],
                                channels="BGR",
                                caption="Uploaded image (no face detected, full image analyzed)",
                                width="stretch",
                            )

                            if res["gradcam"] is not None:
                                st.image(
                                    res["gradcam"],
                                    channels="BGR",
                                    caption="Grad-CAM attention map (full image)",
                                    width="stretch",
                                )

                        # Forensic Tabs: ELA vs Luminance Gradient
                        st.markdown(
                            "<div style='margin-top:15px; font-weight:600;'>"
                            "🔍 Localized Forensic Maps"
                            "</div>",
                            unsafe_allow_html=True
                        )
                        
                        tab_ela, tab_gradient, tab_noise = st.tabs(["⚡ Error Level Analysis (ELA)", "🌊 Luminance Gradient", "🌌 Noise Residuals"])
                        
                        with tab_ela:
                            if res.get("ela_image") is not None:
                                ela_col1, ela_col2 = st.columns([1, 2])
                                with ela_col1:
                                    st.image(
                                        res["ela_image"],
                                        channels="BGR",
                                        caption="ELA map",
                                        width="stretch"
                                    )
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
                            else:
                                st.info("ELA could not be computed for this image format.")
                                
                        with tab_gradient:
                            if res.get("gradient_image") is not None:
                                grad_col1, grad_col2 = st.columns([1, 2])
                                with grad_col1:
                                    st.image(
                                        res["gradient_image"],
                                        channels="BGR",
                                        caption="Luminance Gradient Map",
                                        width="stretch"
                                    )
                                with grad_col2:
                                    st.markdown("**Sobel Gradient Intensity Analysis**")
                                    st.caption(
                                        "High-frequency changes in brightness are highlighted using the Viridis colormap. "
                                        "Inconsistencies, localized blur, or sharp unnatural boundaries typically signal compositing, "
                                        "splicing, or regional generative fill."
                                    )
                            else:
                                st.info("Gradient analysis is unavailable.")

                        with tab_noise:
                            if res.get("noise_residual_image") is not None:
                                noise_col1, noise_col2 = st.columns([1, 2])
                                with noise_col1:
                                    st.image(
                                        res["noise_residual_image"],
                                        channels="BGR",
                                        caption="Noise Residual Map",
                                        width="stretch"
                                    )
                                with noise_col2:
                                    st.markdown("**High-Frequency Noise Residual Map**")
                                    st.caption(
                                        "This isolates high-frequency noise using a median filter subtraction, mapped to the Magma colormap. "
                                        "Natural camera shots show a uniform distribution of noise, whereas spliced, edited, or "
                                        "generative regions typically display distinct local noise anomalies or flat zones."
                                    )
                            else:
                                st.info("Noise residual analysis is unavailable.")

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
                            label_text = f"Camera: {exif.get('make', '')} {exif.get('model', '')}".strip()

                        st.markdown(f"{exif_icon} **{label_text}**")
                        st.caption(exif["suspicion_reason"])

                        if exif["has_exif"] and exif["field_count"]:
                            st.caption(
                                f"{exif['field_count']} EXIF fields present"
                                + (" · GPS data present" if exif["gps_present"] else "")
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

            if batch_errors:
                st.markdown("---")
                st.warning(f"⚠️ {len(batch_errors)} file(s) could not be processed:")

                for fname, reason in batch_errors:
                    st.error(f"**{fname}** — {reason}")

        st.markdown("</div>", unsafe_allow_html=True)

elif analysis_mode == "Forensic Comparison":

    def run_forensic_analysis(uploaded_file):
        """Analyse one image for the comparison view.

        Returns (result_dict, None) on success or (None, error_message).
        Results are cached per content hash so reruns are instant.
        """
        raw_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        cache = st.session_state.setdefault("comparison_cache", {})

        if file_hash not in cache:
            try:
                prediction = predict_image(raw_bytes, temperature=1.0)
            except (PreprocessingError, ValueError) as e:
                return None, f"Could not read this image: {e}"
            except Exception as e:
                logger.error(f"Comparison prediction failed: {e}", exc_info=True)
                return None, "Model inference failed for this image."

            face_image = prediction.get("face_image")
            gradcam_image = None
            try:
                backbone_model = get_backbone_submodel(model)
                last_conv_layer = find_last_conv_layer(backbone_model)
                heatmap = make_gradcam_heatmap(prediction["processed_image"], model, last_conv_layer)
                # The heatmap was computed on the analysed (face) region, so
                # overlay it on that same region.
                gradcam_image = overlay_heatmap(np.ascontiguousarray(face_image), heatmap)
            except Exception as e:
                logger.warning(f"Grad-CAM failed for {uploaded_file.name}: {e}")

            ela_image = compute_ela(raw_bytes)
            cache[file_hash] = {
                "raw_prediction": prediction["raw_prediction"],
                "face_detected": prediction["face_detected"],
                "exif": extract_exif(raw_bytes),
                "ela_image": ela_image,
                "ela_score": ela_uniformity_score(ela_image) if ela_image is not None else None,
                "gradcam": gradcam_image,
                "gradient_image": compute_luminance_gradient(raw_bytes),
                "noise_residual_image": compute_noise_residual(raw_bytes),
            }
            # Keep only the two most recent comparisons in memory.
            while len(cache) > 2:
                cache.pop(next(iter(cache)))

        result = dict(cache[file_hash])
        label, confidence, _ = decode_prediction(
            temperature_scale(result["raw_prediction"], temperature=CALIBRATION_TEMPERATURE)
        )
        result.update(
            filename=uploaded_file.name,
            label=label,
            confidence=confidence,
            is_uncertain=confidence < LOW_CONFIDENCE_THRESHOLD,
        )
        return result, None

    def render_comparison_column(title, uploaded_file, res):
        st.markdown(f"### 🖼 {title}")
        st.image(uploaded_file, width="stretch")
        verdict = res["label"] + (" (uncertain)" if res["is_uncertain"] else "")
        st.metric("Prediction", verdict)
        st.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
        st.caption("Face detected and analysed" if res["face_detected"] else "No face detected; full image analysed")

        if res["gradcam"] is not None:
            st.markdown("#### Grad-CAM")
            st.image(res["gradcam"], channels="BGR", width="stretch")
        if res["ela_image"] is not None:
            st.image(res["ela_image"], channels="BGR", caption="ELA Analysis", width="stretch")
            if res["ela_score"] is not None:
                st.caption(f"ELA uniformity score: {res['ela_score']:.2f} (0 = natural, 1 = AI-like)")
        if res.get("gradient_image") is not None:
            st.image(res["gradient_image"], channels="BGR", caption="Luminance Gradient Map", width="stretch")
        if res.get("noise_residual_image") is not None:
            st.image(res["noise_residual_image"], channels="BGR", caption="Noise Residual Map", width="stretch")

        st.markdown("#### Metadata")
        st.write(res["exif"]["suspicion_reason"])

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🆚 Forensic Image Comparison")

    col1, col2 = st.columns(2)
    with col1:
        image_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png", "webp"], key="compare_a")
    with col2:
        image_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png", "webp"], key="compare_b")

    if model is None and (image_a or image_b):
        st.error("Model could not be loaded. Detection is unavailable.")

    elif image_a and image_b:
        result_a, error_a = run_forensic_analysis(image_a)
        result_b, error_b = run_forensic_analysis(image_b)

        if error_a or error_b:
            if error_a:
                st.error(f"**Image A ({image_a.name})** — {error_a}")
            if error_b:
                st.error(f"**Image B ({image_b.name})** — {error_b}")
        else:
            st.markdown("---")
            left, right = st.columns(2)
            with left:
                render_comparison_column("Image A", image_a, result_a)
            with right:
                render_comparison_column("Image B", image_b, result_b)

            st.markdown("---")
            st.subheader("📊 Comparison Insights")

            confidence_diff = abs(result_a["confidence"] - result_b["confidence"])
            st.metric("Confidence Difference", f"{confidence_diff * 100:.1f}%")

            if result_a["label"] != result_b["label"]:
                st.warning("Classification mismatch detected.")
            else:
                st.success("Both images received the same classification.")

            stronger = result_a if result_a["confidence"] >= result_b["confidence"] else result_b
            st.info(f"{stronger['filename']} has the stronger prediction confidence.")

            # ------------------ EXIF COMPARISON ----------------
            st.markdown("### 🧾 Metadata Comparison")
            exif_a = result_a["exif"]
            exif_b = result_b["exif"]
            fields = [
                ("Has EXIF", "has_exif"),
                ("EXIF Field Count", "field_count"),
                ("Software", "software"),
                ("Camera Make", "make"),
                ("Camera Model", "model"),
                ("Capture Date", "datetime"),
                ("GPS Present", "gps_present"),
                ("AI Software Detected", "ai_software_detected"),
                ("Suspicious", "suspicious"),
            ]
            comparison_rows = [
                {
                    "Field": label_text,
                    # str() keeps the column type uniform for st.dataframe
                    "Image A": str(exif_a.get(key)) if exif_a.get(key) is not None else "N/A",
                    "Image B": str(exif_b.get(key)) if exif_b.get(key) is not None else "N/A",
                }
                for label_text, key in fields
            ]
            st.dataframe(pd.DataFrame(comparison_rows), width="stretch")

            differences = [row["Field"] for row in comparison_rows if row["Image A"] != row["Image B"]]
            if differences:
                st.warning("⚠ Metadata differences detected in: " + ", ".join(differences))
            else:
                st.success("✅ No metadata differences detected.")

            # ------------------ METADATA VERDICT ----------------
            st.markdown("### 🚨 Metadata Verdict")
            for name, exif in (("Image A", exif_a), ("Image B", exif_b)):
                message = f"{name}: {exif.get('suspicion_reason', 'N/A')}"
                if exif.get("suspicious", False):
                    st.warning(message)
                else:
                    st.success(message)

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------- PREDICTION HISTORY / CSV EXPORT --

if st.session_state.get("prediction_history"):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🗂 Prediction History")

    def history_dataframe(rows):
        """Newest-first DataFrame without internal columns."""
        df = pd.DataFrame(list(reversed(rows)))
        return df.drop(columns=["_hash"], errors="ignore")

    if PERSIST_HISTORY:
        st.caption("History is saved to the local SQLite database (PIXELTRUTH_DB_PATH).")
    else:
        st.caption("History is kept for this browser session only (PIXELTRUTH_PERSIST_HISTORY=0).")

    st.dataframe(history_dataframe(st.session_state.prediction_history[-50:]), width="stretch")

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("⬇️ Prepare CSV Report"):
            # Escape every text cell (session rows *and* rows loaded from the
            # database) to block spreadsheet formula injection.
            full_df = history_dataframe(st.session_state.prediction_history).map(sanitize_csv_value)
            st.session_state.prediction_csv = full_df.to_csv(index=False).encode("utf-8")
            st.success("Report prepared — click Download to save the CSV.")

    with c2:
        if st.button("🧹 Clear History"):
            if PERSIST_HISTORY:
                try:
                    clear_history()
                except Exception as e:
                    logger.warning(f"Could not clear history database: {e}", exc_info=True)
            st.session_state.prediction_history = []
            st.session_state.prediction_history_hashes = set()
            st.session_state.prediction_csv = None
            st.success("Prediction history cleared.")

    if st.session_state.get("prediction_csv") is not None:
        st.download_button(
            label="⬇️ Download Report as CSV",
            data=st.session_state.prediction_csv,
            file_name=f"pixeltruth_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- MODEL ANALYTICS ------------------

st.divider()
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📊 Model Analytics Dashboard")
st.caption("Comprehensive performance metrics and visualizations of the deepfake detection model")

cached_metrics = _snapshot_metrics

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
        st.caption(
            "Measured on a held-out test set through the same face-crop pipeline the app uses. "
            "Precision, recall and F1 are support-weighted averages over the Real and Fake classes"
            + (f"; ROC AUC = {cached_metrics['auc']:.4f}." if "auc" in cached_metrics else ".")
        )

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
                width="stretch",
                config={'scrollZoom': True, 'displayModeBar': True}
            )

            st.caption(get_confusion_matrix_caption())

    with col_roc:
        roc_fig = get_roc_curve_plot(cached_metrics)

        if roc_fig:
            st.plotly_chart(
                roc_fig,
                width="stretch",
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
                width="stretch",
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

