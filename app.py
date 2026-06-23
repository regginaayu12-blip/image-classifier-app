import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Image Classification", layout="centered")

st.title("Klasifikasi Gambar Buah")

# 1. Input Link Berbagi Google Drive secara utuh
drive_url = st.text_input(
    "https://drive.google.com/file/d/1WMygo7545PJmH6o0eLl3vglM3NEfF_T9/view?usp=drive_link",
    placeholder="https://drive.google.com/file/d/XXXXX/view?usp=sharing"
)

uploaded_image = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])

img_size = (277, 277)
class_names_text = st.text_input(
    "Nama kelas (pisahkan dengan koma)",
    value="anggur,buah naga"
)

# Fungsi untuk mengunduh model menggunakan URL berbagi
@st.cache_resource
def load_model_from_drive(url):
    output = "temp_model.h5"
    
    if os.path.exists(output):
        os.remove(output)
        
    try:
        with st.spinner("Mengunduh model dari Google Drive... Mohon tunggu."):
            # Menggunakan gdown.download dengan mendeteksi URL secara otomatis
            # gdown akan mengonversi link berbagi menjadi link unduhan langsung
            file_id = gdown.download(url=url, output=output, quiet=False, fuzzy=True)
        
        if file_id and os.path.exists(output):
            model = load_model(output, compile=False)
            return model
        else:
            st.error("Gagal mengunduh file. Pastikan link Google Drive valid dan diatur publik.")
            return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh/memuat model: {e}")
        return None

# Validasi input
if drive_url and uploaded_image:
    class_names = [x.strip() for x in class_names_text.split(",")]

    # Memuat model langsung menggunakan link yang dimasukkan
    model = load_model_from_drive(drive_url)

    if model is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Gambar Uji", use_container_width=True)

        img_resized = img.resize(img_size)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])

        st.subheader("Hasil Prediksi")
        st.write(f"Nilai Output Sigmoid: {confidence:.4f}")

        if confidence > 0.5:
            label = class_names[1] if len(class_names) > 1 else "Kelas 2"
            score = confidence * 100
        else:
            label = class_names[0] if len(class_names) > 0 else "Kelas 1"
            score = (1 - confidence) * 100

        st.success(f"Prediksi: {label.upper()}")
        st.write(f"Confidence: {score:.2f}%")
