import streamlit as st

st.set_page_config(page_title="PixelTruth", layout="wide")

st.title("🔍 PixelTruth - Deepfake Detector")
st.write("Testing - if you see this, the app works!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded")
    st.success("Image uploaded!")