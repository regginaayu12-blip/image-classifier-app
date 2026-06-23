import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import os
import zipfile
import pandas as pd
import shutil

st.set_page_config(page_title="Image Classification Batch", layout="centered")

st.title("Klasifikasi Gambar Buah (Batch .zip)")

# 1. Input Link Berbagi Google Drive
drive_url = st.text_input(
    "https://drive.google.com/file/d/1WMygo7545PJmH6o0eLl3vglM3NEfF_T9/view?usp=drive_link",
    placeholder="https://drive.google.com/file/d/XXXXX/view?usp=sharing"
)

# 2. Mengubah input menjadi file .zip
uploaded_zip = st.file_uploader("Upload file gambar dalam format (.zip)", type=["zip"])

img_size = (277, 277)
class_names_text = st.text_input(
    "Nama kelas (pisahkan dengan koma)",
    value="anggur,buah naga"
)

# Fungsi untuk mengunduh model
@st.cache_resource
def load_model_from_drive(url):
    output = "temp_model.h5"
    if os.path.exists(output):
        os.remove(output)
        
    try:
        with st.spinner("Mengunduh model dari Google Drive..."):
            file_id = gdown.download(url=url, output=output, quiet=False)
        
        if file_id and os.path.exists(output):
            # Menambahkan compile=False dan custom_objects/safe_mode untuk meredam error Keras versi baru
            try:
                model = load_model(output, compile=False)
            except Exception:
                model = tf.keras.models.load_model(output, compile=False, safe_mode=False)
            return model
        else:
            st.error("Gagal mengunduh file. Pastikan link Google Drive valid.")
            return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh/memuat model: {e}")
        return None

# Eksekusi utama jika link drive dan file zip tersedia
if drive_url and uploaded_zip:
    class_names = [x.strip() for x in class_names_text.split(",")]

    # Memuat model
    model = load_model_from_drive(drive_url)

    if model is not None:
        # Folder temporer untuk mengekstrak file zip
        extract_dir = "temp_extracted_images"
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir)

        # Mengekstrak file ZIP
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Mencari semua file gambar di dalam folder hasil ekstraksi
        valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        all_images = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    all_images.append(os.path.join(root, file))

        if len(all_images) == 0:
            st.warning("Tidak ditemukan file gambar (.jpg, .jpeg, .png) di dalam file .zip tersebut.")
        else:
            st.info(f"Berhasil menemukan {len(all_images)} gambar di dalam file ZIP. Memulai prediksi...")
            
            # List untuk menampung hasil tabel ringkasan
            results = []

            # Melakukan prediksi untuk setiap gambar
            for img_path in all_images:
                filename = os.path.basename(img_path)
                try:
                    # Membuka dan preprocess gambar
                    img = Image.open(img_path).convert("RGB")
                    img_resized = img.resize(img_size)
                    img_array = np.array(img_resized, dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    prediction = model.predict(img_array, verbose=0)
                    confidence = float(prediction[0][0])

                    # Logika penentuan kelas
                    if confidence > 0.5:
                        label = class_names[1] if len(class_names) > 1 else "Kelas 2"
                        score = confidence * 100
                    else:
                        label = class_names[0] if len(class_names) > 0 else "Kelas 1"
                        score = (1 - confidence) * 100

                    results.append({
                        "Nama File": filename,
                        "Hasil Prediksi": label.upper(),
                        "Confidence": f"{score:.2f}%",
                        "Nilai Raw Sigmoid": f"{confidence:.4f}"
                    })
                except Exception as e:
                    results.append({
                        "Nama File": filename,
                        "Hasil Prediksi": "ERROR",
                        "Confidence": "0%",
                        "Nilai Raw Sigmoid": str(e)
                    })

            # Menampilkan hasil akhir berupa tabel ringkasan
            st.subheader("📋 Ringkasan Hasil Klasifikasi")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            # Tampilkan gambar beserta hasil prediksinya satu-satu di bawah tabel (Opsional)
            st.subheader("🖼️ Detail Visual Gambar")
            for img_path, res in zip(all_images, results):
                if res["Hasil Prediksi"] != "ERROR":
                    img = Image.open(img_path)
                    st.image(img, caption=f"{res['Nama File']} -> {res['Hasil Prediksi']} ({res['Confidence']})", width=300)
                    st.markdown("---")

        # Bersihkan folder sampah setelah selesai proses
        shutil.rmtree(extract_dir)
