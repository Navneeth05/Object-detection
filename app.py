import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="YOLO Animal Detection", layout="wide")

st.title("🐾 YOLO Animal Detection System")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Sidebar options
option = st.sidebar.selectbox(
    "Choose Detection Mode",
    ["Upload Image", "Live Camera"]
)

# ---------------- UPLOAD IMAGE ---------------- #
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        results = model(image_np)
        annotated_frame = results[0].plot()

        st.image(annotated_frame, caption="Detection Result", use_container_width=True)

# ---------------- LIVE CAMERA ---------------- #
elif option == "Live Camera":
    st.info("Click Start to use webcam")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Camera not working")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        FRAME_WINDOW.image(annotated_frame, channels="BGR")

    camera.release()
