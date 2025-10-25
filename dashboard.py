import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import tempfile
import pandas as pd

# ============================
# Folder model & config
# ============================
MODEL_DIR = "model_uts"
os.makedirs(MODEL_DIR, exist_ok=True)
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "NadiaLaporan4.pt")
YOLO_DEFAULT_URL = "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8n.pt"

# ============================
# Import YOLO
# ============================
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ============================
# Helper functions
# ============================
def download_file(url, dest_path):
    """Download file dari URL jika tidak ada di path."""
    st.info(f"Mendownload file dari {url} ...")
    r = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
    st.success(f"File tersimpan di {dest_path}")

@st.cache_resource
def load_yolo_model(path):
    """Load YOLO model, download jika tidak ada."""
    if YOLO is None:
        st.error("Ultralytics YOLO belum terinstal. Jalankan 'pip install ultralytics'.")
        return None
    if not os.path.exists(path):
        st.info("File YOLO tidak ditemukan. Mengunduh YOLOv8 default ...")
        download_file(YOLO_DEFAULT_URL, path)
    return YOLO(path)

def run_yolo_inference(model, img, conf_thresh=0.45):
    """Inferensi YOLO pada PIL Image."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name
    try:
        results = model(tmp_path, conf=conf_thresh)
        plotted = results[0].plot()
        return results, Image.fromarray(plotted)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

def extract_detections(results):
    """Ekstrak hasil deteksi menjadi DataFrame."""
    det_list = []
    boxes = getattr(results[0], "boxes", None)
    names = getattr(results[0], "names", None)
    if boxes:
        for b in boxes:
            cls = int(getattr(b, "cls")) if hasattr(b, "cls") else None
            conf = float(getattr(b, "conf")) if hasattr(b, "conf") else None
            xy = getattr(b, "xyxy", [None, None, None, None])
            label = names[cls] if names and cls is not None else f"class_{cls}"
            det_list.append({
                "label": label,
                "class_id": cls,
                "confidence": conf,
                "x1": xy[0],
                "y1": xy[1],
                "x2": xy[2],
                "y2": xy[3]
            })
    return pd.DataFrame(det_list) if det_list else None

# ============================
# UI Styles
# ============================
def set_gradient():
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #fff8e6 0%, #e6f4ea 100%);
        }
        </style>
    """, unsafe_allow_html=True)

set_gradient()
st.set_page_config(page_title="🍽 YOLO Food Detection Dashboard", layout="wide")
st.title("🍽 YOLO Food Detection Dashboard")

# ============================
# Sidebar
# ============================
st.sidebar.header("⚙ Pengaturan Deteksi")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.01)

# ============================
# Upload Image
# ============================
uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg","jpeg","png"])
if uploaded_food:
    img = Image.open(uploaded_food).convert("RGB")
    st.image(img, caption="Gambar Diupload", width='stretch')

    # Load YOLO model
    model_yolo = load_yolo_model(YOLO_MODEL_PATH)
    if model_yolo:
        # Run inference
        results, plotted_img = run_yolo_inference(model_yolo, img, conf_thresh)
        st.image(plotted_img, caption="Hasil Deteksi", width='stretch')

        # Extract detections
        df = extract_detections(results)
        if df is not None and not df.empty:
            st.subheader("📋 Daftar Deteksi")
            st.dataframe(df)
            st.subheader("📊 Ringkasan Kategori")
            st.bar_chart(df["label"].value_counts())
            st.download_button(
                "⬇ Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "detection_results.csv",
                "text/csv"
            )
        else:
            st.warning("Tidak ada objek terdeteksi di atas threshold.")
