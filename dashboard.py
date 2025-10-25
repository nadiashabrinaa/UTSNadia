
import requests
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import os

# ---------------------------
# Imports: TensorFlow & YOLO
# ---------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------------------
# Fungsi update warna & tombol
# ---------------------------
def update_colors(mode, is_home=False):
    if mode == "Klasifikasi Penyakit Daun Teh":
        bg_color = "#e6f4ea"       # soft hijau
        sidebar_color = "#d9f0d3"
        btn_color = "#4caf50"
        btn_hover = "#45a049"
    else:
        bg_color = "#fff8e6"       # soft krem
        sidebar_color = "#fff3d9"
        btn_color = "#f4b400"
        btn_hover = "#f2a900"

    # Gradient untuk halaman home
    if is_home:
        bg_style = "linear-gradient(135deg, #e6f4ea 0%, #fff8e6 100%)"
    else:
        bg_style = bg_color

    st.markdown(f"""
        <style>
        div[data-testid="stAppViewContainer"] {{
            background: {bg_style} !important;
        }}
        div[data-testid="stSidebar"] {{
            background-color: {sidebar_color} !important;
        }}
        button {{
            background-color: {btn_color} !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px 20px;
            box-shadow: 2px 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }}
        button:hover {{
            background-color: {btn_hover} !important;
            transform: scale(1.05);
        }}
        .stButton button {{
            width: 100% !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Helper: download file otomatis
# ---------------------------
def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        st.info(f"Mendownload model YOLO dari {url} ...")
        response = requests.get(url, stream=True)
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
        st.success(f"Model tersimpan di {dest_path}")

# ---------------------------
# Config page
# ---------------------------
st.set_page_config(page_title="AI Vision Dashboard", layout="wide")

# ---------------------------
# Session state
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "mode" not in st.session_state:
    st.session_state.mode = "Klasifikasi Penyakit Daun Teh"

# =====================================================
# Halaman Pembuka
# =====================================================
if st.session_state.page == "home":
    # Update warna halaman home dengan gradient
    update_colors(st.session_state.mode, is_home=True)

    st.markdown(
        "<h1 style='text-align:center; font-size:42px;'>🤖 Selamat Datang di <b>AI Vision Dashboard</b></h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size:18px;'>"
        "Sebuah sistem berbasis <b>Kecerdasan Buatan (AI)</b> untuk "
        "<b>klasifikasi penyakit daun teh</b> dan <b>deteksi jenis makanan</b> secara otomatis dan interaktif."
        "</p>",
        unsafe_allow_html=True
    )
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712105.png",
        width=250,
        caption="AI Vision System — Powered by Streamlit, TensorFlow & YOLOv8"
    )
    st.markdown("---")
    st.markdown("### Kelebihan Dashboard")
    st.markdown("""
    - ⚡ *Cepat & Akurat* – Proses gambar hanya dalam hitungan detik  
    - 🧠 *Ditenagai AI Modern* – Menggunakan model CNN & YOLOv8  
    - 🌿 *Dua Fungsi Utama* – Analisis daun teh & deteksi jenis makanan  
    - 📊 *Interaktif & Informatif* – Hasil tampil otomatis dalam grafik  
    - ☁ *Ramah Pengguna* – Tidak perlu instalasi tambahan, cukup unggah gambar  
    """)
    st.markdown("---")
    st.markdown(""" Dikembangkan Oleh: Nadia Shabrina | Universitas Syiah Kuala
    """)
    st.markdown("---")
    st.markdown("### Kegunaan Dashboard")
    st.markdown("""
    Dashboard ini dirancang untuk membantu penelitian, pembelajaran, dan demonstrasi teknologi
    *Computer Vision*.  
    Pengguna dapat:
    - 🔍 Mengidentifikasi penyakit pada daun teh berdasarkan citra  
    - 🍽 Mendeteksi jenis makanan (Meal, Dessert, Drink) secara otomatis  
    - 💾 Menyimpan hasil deteksi ke file *CSV* untuk analisis lanjutan  
    """)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Mulai Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    st.stop()

# ---------------------------
# Tombol kembali
# ---------------------------
st.markdown("""
    <style>
    .back-button {
        position: fixed;
        top: 20px;
        left: 20px;
        background-color: #f0f2f6;
        color: #333;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        font-size: 22px;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .back-button:hover {
        background-color: #e0e0e0;
        transform: scale(1.1);
    }
    </style>
    <form action="#" method="post">
        <button class="back-button" name="back" type="submit">⬅</button>
    </form>
""", unsafe_allow_html=True)

if st.session_state.get("page") == "dashboard":
    if st.button("⬅ Kembali ke Halaman Utama", key="back_button_hidden"):
        st.session_state.page = "home"

# =====================================================
# Sidebar + Mode + Warna
# =====================================================
with st.sidebar:
    st.header("⚙ Pilih Mode")
    mode = st.radio("Mode Analisis:", ["Klasifikasi Penyakit Daun Teh", "Deteksi Jenis Makanan"])
    conf_thresh = st.slider("Confidence Threshold (untuk YOLO)", 0.1, 1.0, 0.45, 0.01)
    st.markdown("---")

# Simpan pilihan ke session_state
st.session_state.mode = mode

# Update warna dashboard (solid)
update_colors(st.session_state.mode, is_home=False)

# =====================================================
# Mulai dashboard asli
# =====================================================
st.markdown("<h1 style='text-align:center;'>🤖 AI Vision Dashboard</h1>", unsafe_allow_html=True)
st.write("Klasifikasi Penyakit Daun Teh  | Deteksi Jenis Makanan")

# =====================================================
# Model paths dan classes
# =====================================================
POSSIBLE_TEA_PATHS = ["model_uts/nadiashabrinaLaporan2.h5","nadiashabrinaLaporan2.h5"]
POSSIBLE_FOOD_PATHS = ["model_uts/NadiaLaporan4.pt"]

def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

MODEL_TEA_PATH = find_existing_path(POSSIBLE_TEA_PATHS)
os.makedirs("model_uts", exist_ok=True)

YOLO_URL = "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolov8n.pt"
MODEL_FOOD_PATH = find_existing_path(POSSIBLE_FOOD_PATHS)
if MODEL_FOOD_PATH is None:
    MODEL_FOOD_PATH = "model_uts/yolov8n.pt"
    download_file(YOLO_URL, MODEL_FOOD_PATH)

TEA_CLASSES = ["Red Leaf Spot","Algal Leaf Spot","Bird’s Eyespot","Gray light","White Spot","Anthracnose","Brown Blight","Healthy Tea Leaves"]
FOOD_CLASSES = ["Meal","Dessert","Drink"]

# =====================================================
# Helpers: load models & preprocessing
# =====================================================
@st.cache_resource
def load_keras_model_safe(path):
    if tf is None:
        raise RuntimeError("TensorFlow belum terinstal di environment.")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File model Keras tidak ditemukan: {path}")
    return load_model(path)

@st.cache_resource
def load_yolo_model_safe(path):
    if YOLO is None:
        raise RuntimeError("Ultralytics (YOLO) belum terinstal.")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File model YOLO tidak ditemukan: {path}")
    return YOLO(path)

def safe_softmax(x):
    x = np.array(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()

def preprocess_for_keras(pil_image, model):
    if model is None:
        raise RuntimeError("Model Keras belum dimuat.")
    input_shape = getattr(model, "input_shape", None)
    if not input_shape:
        raise RuntimeError("Tidak dapat membaca input_shape dari model.")
    if len(input_shape) == 2:
        raise RuntimeError("Model tampak menerima input 1D — skrip ini mengharapkan model citra (H,W,3).")
    h, w = input_shape[1], input_shape[2]
    if h is None or w is None:
        h, w = 224, 224
    img = pil_image.resize((w, h))
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

# =====================================================
# Main UI logic
# =====================================================
if mode == "Klasifikasi Penyakit Daun Teh":
    st.subheader("Klasifikasi Penyakit Daun Teh Berdasarkan Citra")
    uploaded_img = st.file_uploader("Unggah gambar daun teh", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)
        try:
            if MODEL_TEA_PATH is None:
                st.error("File model .h5 tidak ditemukan.")
            else:
                model_tea = load_keras_model_safe(MODEL_TEA_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model Keras: {e}")
            model_tea = None

        if model_tea:
            try:
                arr = preprocess_for_keras(image, model_tea)
                preds = model_tea.predict(arr)
                if preds.ndim > 1:
                    preds = preds[0]
                if preds.max() > 1.0 or preds.min() < 0:
                    preds = safe_softmax(preds)
                top_idx = int(np.argmax(preds))
                label = TEA_CLASSES[top_idx] if top_idx < len(TEA_CLASSES) else f"Class {top_idx}"
                conf = float(preds[top_idx])
                st.success(f"Prediksi: {label} (Confidence: {conf:.3f})")
                df = pd.DataFrame({"Class": TEA_CLASSES, "Confidence": np.round(preds, 4)})
                st.dataframe(df)
                st.bar_chart(df.set_index("Class"))
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

else:  # Deteksi makanan
    st.subheader("Deteksi Jenis Makanan (Meal, Dessert, Drink)")
    uploaded_food = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png"])
    if uploaded_food:
        image = Image.open(uploaded_food).convert("RGB")
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        # Load YOLO model
        try:
            if YOLO is None:
                st.warning("Ultralytics (YOLO) belum terinstal.")
                model_yolo = None
            else:
                model_yolo = load_yolo_model_safe(MODEL_FOOD_PATH)
        except Exception as e:
            st.warning(f"YOLO tidak tersedia: {e}")
            model_yolo = None

        if model_yolo:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                results = model_yolo(tmp_path, conf=conf_thresh)
                plotted = results[0].plot()
                result_pil = Image.fromarray(np.array(plotted)) if not isinstance(plotted, np.ndarray) else Image.fromarray(plotted)
                st.image(result_pil, caption="Hasil Deteksi", use_container_width=True)

                det_rows = []
                names = getattr(results[0], "names", None)
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None:
                    for b in boxes:
                        try:
                            cls = int(getattr(b, "cls").item() if hasattr(b, "cls") and hasattr(b.cls, "item") else getattr(b, "cls"))
                        except Exception:
                            cls = None
                        try:
                            conf = float(getattr(b, "conf").item() if hasattr(b, "conf") and hasattr(b.conf, "item") else getattr(b, "conf"))
                        except Exception:
                            conf = None
                        try:
                            xyxy = getattr(b, "xyxy")
                            xy = xyxy.cpu().numpy().tolist() if hasattr(xyxy, "cpu") else np.array(xyxy).tolist()
                        except Exception:
                            xy = [None, None, None, None]
                        label = names[int(cls)] if (names is not None and cls is not None and int(cls) in names) else (f"class_{cls}" if cls is not None else "")
                        det_rows.append({
                            "label": label,
                            "class_id": cls,
                            "confidence": conf,
                            "x1": xy[0] if len(xy) >= 1 else None,
                            "y1": xy[1] if len(xy) >= 2 else None,
                            "x2": xy[2] if len(xy) >= 3 else None,
                            "y2": xy[3] if len(xy) >= 4 else None,
                        })

                if det_rows:
                    df = pd.DataFrame(det_rows)
                    st.subheader("📋Daftar Objek Terdeteksi")
                    st.dataframe(df)
                    st.subheader("📊Ringkasan Kategori")
                    st.bar_chart(df["label"].value_counts())
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download hasil (CSV)", csv, "detection_results.csv", "text/csv")
                else:
                    st.warning("Tidak ada objek terdeteksi di atas threshold.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat inferensi YOLO: {e}")
            finally:
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        else:
            st.info("Fungsi deteksi YOLO tidak tersedia — periksa apakah paket ultralytics terinstal dan file .pt berada di repo.")
