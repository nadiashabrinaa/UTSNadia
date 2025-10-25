import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import tempfile
import pandas as pd

# ================================
# Setup folder model
# ================================
MODEL_DIR = "Model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# Import libraries optional
# ================================
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ================================
# Download helper
# ================================
def download_file(url, dest_path):
    st.info(f"Mendownload file dari {url} ...")
    r = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
    st.success(f"File tersimpan di {dest_path}")

# ================================
# File paths
# ================================
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "nadiashabrinaLaporan2.h5")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "NadiaLaporan4.pt")
YOLO_DEFAULT_URL = "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8n.pt"

# ================================
# Load models with caching
# ================================
@st.cache_resource
def load_keras_model(path):
    if tf is None:
        st.error("TensorFlow belum terinstal.")
        return None
    if not os.path.exists(path):
        st.warning(f"File model Keras tidak ditemukan: {path}")
        return None
    return load_model(path)

@st.cache_resource
def load_yolo_model(path):
    if YOLO is None:
        st.warning("Ultralytics YOLO belum terinstal.")
        return None
    if not os.path.exists(path):
        st.info("File YOLO tidak ditemukan. Mengunduh model YOLOv8 default ...")
        download_file(YOLO_DEFAULT_URL, path)
    return YOLO(path)

# ================================
# Helper preprocessing
# ================================
def preprocess_image(img, target_size=(224,224)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="AI Vision Dashboard", layout="wide")
st.title("🤖 AI Vision Dashboard")

# Sidebar
mode = st.sidebar.radio("Pilih Mode:", ["Klasifikasi Penyakit Daun Teh", "Deteksi Jenis Makanan"])
conf_thresh = st.sidebar.slider("Confidence YOLO", 0.1, 1.0, 0.45, 0.01)

# ================================
# Klasifikasi Daun Teh
# ================================
TEA_CLASSES = ["Red Leaf Spot","Algal Leaf Spot","Bird’s Eyespot","Gray light","White Spot","Anthracnose","Brown Blight","Healthy Tea Leaves"]

if mode == "Klasifikasi Penyakit Daun Teh":
    st.subheader("Klasifikasi Daun Teh")
    uploaded_img = st.file_uploader("Unggah gambar daun teh", type=["jpg","jpeg","png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Gambar Diupload", use_container_width=True)
        model = load_keras_model(KERAS_MODEL_PATH)
        if model:
            arr = preprocess_image(img, target_size=model.input_shape[1:3])
            preds = model.predict(arr)
            if preds.ndim > 1:
                preds = preds[0]
            if preds.max() > 1.0 or preds.min() < 0:
                preds = softmax(preds)
            top_idx = int(np.argmax(preds))
            st.success(f"Prediksi: {TEA_CLASSES[top_idx]} (Confidence: {preds[top_idx]:.3f})")
            df = pd.DataFrame({"Class": TEA_CLASSES, "Confidence": np.round(preds,4)})
            st.dataframe(df)
            st.bar_chart(df.set_index("Class"))

# ================================
# Deteksi Makanan
# ================================
FOOD_CLASSES = ["Meal","Dessert","Drink"]

if mode == "Deteksi Jenis Makanan":
    st.subheader("Deteksi Jenis Makanan")
    uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg","jpeg","png"])
    if uploaded_food:
        img = Image.open(uploaded_food).convert("RGB")
        st.image(img, caption="Gambar Diupload", use_container_width=True)
        model_yolo = load_yolo_model(YOLO_MODEL_PATH)
        if model_yolo:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            results = model_yolo(tmp_path, conf=conf_thresh)
            plotted = results[0].plot()
            st.image(Image.fromarray(plotted), caption="Hasil Deteksi", use_container_width=True)
            # Buat dataframe hasil deteksi
            det_list = []
            boxes = getattr(results[0], "boxes", None)
            names = getattr(results[0], "names", None)
            if boxes:
                for b in boxes:
                    cls = int(getattr(b,"cls")) if hasattr(b,"cls") else None
                    conf = float(getattr(b,"conf")) if hasattr(b,"conf") else None
                    xy = getattr(b,"xyxy", [None,None,None,None])
                    label = names[cls] if names and cls is not None else f"class_{cls}"
                    det_list.append({
                        "label": label, "class_id": cls, "confidence": conf,
                        "x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3]
                    })
            if det_list:
                df = pd.DataFrame(det_list)
                st.subheader("📋Daftar Deteksi")
                st.dataframe(df)
                st.bar_chart(df["label"].value_counts())
                st.download_button("⬇ Download CSV", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")
            try:
                os.remove(tmp_path)
            except: pass
