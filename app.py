import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from gradcam import make_gradcam_heatmap, overlay_heatmap
from metrics import (
    get_sample_metrics,
    get_confusion_matrix_plot,
    get_roc_curve_plot,
    get_dataset_distribution_plot,
    get_class_statistics,
    get_confusion_matrix_caption,
    get_roc_curve_caption,
    get_dataset_distribution_caption,
    plot_confidence_bars,
    plot_live_confidence_gauge,
)
from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256

st.set_page_config(page_title="PixelTruth", page_icon="🔍", layout="wide")

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

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict_image(image):
    if model is None:
        return None, None, None, None, None
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    class_label = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    real_prob = float(prediction[0][0])
    fake_prob = float(prediction[0][1])
    label = "Real" if class_label == 0 else "Fake"
    return label, confidence, real_prob, fake_prob, processed_image

st.title("🔍 DEEPFAKE SENTINEL")
st.markdown("AI-powered detection of manipulated social media images")

st.divider()

col_info_left, col_info_right = st.columns([2, 1])
with col_info_left:
    st.subheader("🧠 Understanding Deepfakes")
    st.markdown("""
    - **Deepfakes** are AI-generated images where faces/content are manipulated
    - They can be used for **misinformation**, **fraud**, and **privacy attacks**
    - Detection models analyze subtle artifacts in lighting, edges, and facial structure
    """)
with col_info_right:
    st.subheader("📈 Model Info")
    st.metric("Training Accuracy", "95%")
    st.metric("Input Size", "96 × 96 px")
    st.metric("Task", "Binary Classification")

st.divider()
st.header("🖼 Upload an Image")

uploaded_file = st.file_uploader(
    "Drop or browse a social media image",
    type=["jpg", "jpeg", "png", "webp"],
)

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"File too large: {uploaded_file.size / (1024 * 1024):.1f} MB")
        image = None
    else:
        try:
            raw_bytes = uploaded_file.read()
            file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
            uploaded_file.seek(0)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Invalid image file")
                image = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            image = None

    if image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, channels="BGR", width=400)
        
        with col2:
            st.subheader("📊 Detection Result")
            
            if model is None:
                st.error("Model not loaded")
            else:
                with st.spinner("Analyzing..."):
                    label, confidence, real_prob, fake_prob, processed_image = predict_image(image)

                if label is not None:
                    icon = "🟢" if label == "Real" else "🔴"
                    headline = "AUTHENTIC IMAGE" if label == "Real" else "DEEPFAKE SUSPECTED"
                    
                    if label == "Real":
                        st.success(f"{icon} **{headline}**")
                    else:
                        st.error(f"{icon} **{headline}**")
                    
                    st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                    
                    st.pyplot(plot_live_confidence_gauge(real_prob, fake_prob), width='stretch')
                    
                    col_real, col_fake = st.columns(2)
                    with col_real:
                        st.metric("Real", f"{real_prob*100:.1f}%")
                        st.progress(real_prob)
                    with col_fake:
                        st.metric("Fake", f"{fake_prob*100:.1f}%")
                        st.progress(fake_prob)

                    st.pyplot(plot_confidence_bars(real_prob, fake_prob), width='stretch')

                    uncertainty_threshold = 0.15
                    is_borderline = abs(real_prob - fake_prob) < uncertainty_threshold
                    
                    if is_borderline:
                        st.warning(f"⚠️ Uncertain: Real: {real_prob*100:.1f}% | Fake: {fake_prob*100:.1f}%")
                    elif confidence < 0.6:
                        st.info(f"Low Confidence: {confidence*100:.1f}%")

                    if label == "Fake":
                        st.info("Patterns consistent with deepfake artifacts detected")
                    else:
                        st.info("No significant deepfake indicators detected")

                    try:
                        conv_layers = [l.name for l in model.layers if 'conv' in l.name.lower()]
                        if conv_layers:
                            last_conv_layer = conv_layers[-1]
                            heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer)
                            gradcam_image = overlay_heatmap(image, heatmap)
                            
                            st.divider()
                            st.subheader("🧠 Grad-CAM Visualization")
                            col_orig, col_gcam = st.columns(2)
                            with col_orig:
                                st.image(image, channels="BGR", caption="Original", width='stretch')
                            with col_gcam:
                                st.image(gradcam_image, channels="BGR", caption="Heatmap", width='stretch')
                    except:
                        pass

st.divider()

st.header("📈 Training Performance")
if os.path.exists("Figure_2.png") and os.path.exists("Figure_1.png"):
    col1, col2 = st.columns(2)
    with col1:
        st.image("Figure_2.png", caption="Accuracy", width='stretch')
    with col2:
        st.image("Figure_1.png", caption="Loss", width='stretch')
else:
    st.info("Training graphs not found")

st.divider()

st.header("📊 Model Analytics Dashboard")

metrics = get_sample_metrics()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
col2.metric("Precision", f"{metrics['precision']:.1f}%")
col3.metric("Recall", f"{metrics['recall']:.1f}%")
col4.metric("F1-Score", f"{metrics['f1_score']:.1f}%")

st.divider()

col_cm, col_roc = st.columns(2)
with col_cm:
    st.pyplot(get_confusion_matrix_plot(), width='stretch')
    st.caption(get_confusion_matrix_caption())
with col_roc:
    st.pyplot(get_roc_curve_plot(), width='stretch')
    st.caption(get_roc_curve_caption())

st.divider()

col_dist, col_stats = st.columns(2)
with col_dist:
    st.pyplot(get_dataset_distribution_plot(), width='stretch')
    st.caption(get_dataset_distribution_caption())
with col_stats:
    st.markdown("**Per-Class Performance**")
    for class_label, stats in get_class_statistics().items():
        icon = "🟢" if class_label == "Real" else "🔴"
        st.markdown(f"**{icon} {class_label}:** {stats['correctly_classified']:,}/{stats['total_samples']:,} ({stats['class_accuracy']:.1f}%)")

st.markdown("---")
st.markdown("🕵️ **PixelTruth** • Built with Streamlit & TensorFlow")