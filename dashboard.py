import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import tempfile
import pandas as pd

# YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ================================
# Load Models
# ================================
@st.cache_resource
def load_models():
    yolo_model, classifier = None, None
    # Load YOLO
    if YOLO_AVAILABLE:
        yolo_path = "Model/NadiaLaporan4.pt"
        if os.path.exists(yolo_path):
            try:
                yolo_model = YOLO(yolo_path)
            except Exception as e:
                st.warning(f"Gagal memuat YOLO: {e}")
        else:
            st.warning(f"File YOLO tidak ditemukan di {yolo_path}")
    else:
        st.warning("Ultralytics YOLO belum terinstal. Deteksi makanan tidak tersedia.")

    # Load Keras classifier
    classifier_path = "Model/nadiashabrinaLaporan2.h5"
    if os.path.exists(classifier_path):
        try:
            classifier = load_model(classifier_path)
        except Exception as e:
            st.error(f"Gagal memuat model Keras: {e}")
    else:
        st.warning(f"File Keras tidak ditemukan di {classifier_path}")

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ================================
# Helper: preprocess Keras
# ================================
def preprocess_for_keras(pil_image, model):
    input_shape = getattr(model, "input_shape", (None, 224, 224, 3))
    h, w = input_shape[1] or 224, input_shape[2] or 224
    img = pil_image.resize((w, h))
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

# ================================
# Main UI
# ================================
st.title("🤖 AI Vision Dashboard")

mode = st.radio("Pilih Mode:", ["Klasifikasi Penyakit Daun Teh", "Deteksi Jenis Makanan"])

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Diupload", use_container_width=True)

    if mode == "Klasifikasi Penyakit Daun Teh":
        if classifier:
            arr = preprocess_for_keras(img, classifier)
            preds = classifier.predict(arr)[0]
            top_idx = int(np.argmax(preds))
            st.success(f"Prediksi kelas: {top_idx} (Confidence: {preds[top_idx]:.3f})")
        else:
            st.warning("Model Keras belum tersedia.")

    else:  # Deteksi makanan
        if yolo_model:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img.save(tmp_file.name)
            results = yolo_model(tmp_file.name)
            plotted = results[0].plot()
            st.image(plotted, caption="Hasil Deteksi", use_container_width=True)
            os.remove(tmp_file.name)
        else:
            st.warning("Model YOLO tidak tersedia atau belum diinstal.")
