import streamlit as st

st.set_page_config(page_title="PixelTruth", layout="wide")

st.title("🔍 PixelTruth - Deepfake Detector")
st.write("App is working!")

st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.success("Image uploaded successfully!")

st.header("Model Analytics")
st.info("Model analytics would appear here")