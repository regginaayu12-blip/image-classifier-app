import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

# Set up the page title and design
st.set_page_config(page_title="Aplikasi Klasifikasi Buah CNN", layout="centered")
st.title("🍓 Aplikasi Klasifikasi Gambar Buah (CNN)")
st.write("Ubah notebook klasifikasi menjadi aplikasi interaktif menggunakan Streamlit.")

# =========================================================
# CONFIGURATION
# =========================================================
# Anda dapat menyesuaikan path model di bawah ini jika diletakkan di folder berbeda
DEFAULT_MODEL_PATH = "model_buah_cnn.keras"
IMG_SIZE = (277, 277)
CLASS_NAMES = ['anggur', 'buah naga']

# Sidebar untuk upload model atau konfigurasi
st.sidebar.header("⚙️ Pengaturan Model")
model_file_path = st.sidebar.text_input("Path Model (.keras):", value=DEFAULT_MODEL_PATH)

# Fungsi untuk memuat model dengan cache agar tidak reload terus menerus
@st.cache_resource
def load_cnn_model(path):
    if os.path.exists(path):
        try:
            model = load_model(path)
            return model, f"[INFO] Model berhasil dimuat dari {path}!"
        except Exception as e:
            return None, f"[ERROR] Gagal memuat model: {str(e)}"
    else:
        return None, f"[ERROR] File model tidak ditemukan di '{path}'. Pastikan file model sudah di-upload ke server/direktori."

# Load model secara otomatis jika file tersedia
model, message = load_cnn_model(model_file_path)

if model is not None:
    st.sidebar.success(message)
    
    st.write("### 📤 Upload Gambar Buah untuk Diprediksi")
    st.info("Aplikasi ini mendukung klasifikasi untuk kelas: **" + ", ".join([c.upper() for c in CLASS_NAMES]) + "**")
    
    # Widget upload file gambar (bisa multi file atau single file)
    uploaded_files = st.file_uploader("Pilih gambar buah (.png, .jpg, .jpeg)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Menampilkan **{len(uploaded_files)}** gambar yang di-upload:")
        
        # Loop semua gambar yang di-upload
        for uploaded_file in uploaded_files:
            st.write("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Load gambar menggunakan PIL untuk ditampilkan di streamlit
                img_pil = Image.open(uploaded_file)
                st.image(img_pil, caption=f"Gambar Uji: {uploaded_file.name}", use_container_width=True)
                
            with col2:
                st.write(f"**Nama File:** `{uploaded_file.name}`")
                
                with st.spinner("Sedang melakukan prediksi..."):
                    # Preprocessing Gambar sesuai dengan spesifikasi model notebook asli
                    # Resize gambar ke target_size (277, 277)
                    img_resized = img_pil.resize(IMG_SIZE)
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
                    
                    # Prediksi menggunakan model CNN
                    prediction = model.predict(img_array)
                    confidence = prediction[0][0]
                    
                    st.write("#### **Hasil Analisis Klasifikasi:**")
                    # Klasifikasi Binary (Sigmoid) sesuai logic notebook asli
                    if confidence > 0.5:
                        predicted_class = CLASS_NAMES[1].upper()
                        conf_score = confidence * 100
                        st.success(f"**Prediksi:** {predicted_class}")
                        st.metric(label="Confidence Score", value=f"{conf_score:.2f}%")
                    else:
                        predicted_class = CLASS_NAMES[0].upper()
                        conf_score = (1 - confidence) * 100
                        st.success(f"**Prediksi:** {predicted_class}")
                        st.metric(label="Confidence Score", value=f"{conf_score:.2f}%")
else:
    st.sidebar.error(message)
    st.warning("⚠️ Silakan pastikan file model `.keras` Anda berada pada path yang benar di sidebar agar aplikasi dapat berfungsi.")
    st.write("### Cara Menjalankan Aplikasi ini Secara Lokal:")
    st.write("1. Simpan script ini dengan nama `streamlit_app.py`.")
    st.write("2. Pastikan file model Anda (`model_buah_cnn.keras`) berada di folder yang sama atau sesuaikan path di sidebar.")
    st.write("3. Install library pendukung melalui terminal:")
    st.code("pip install streamlit tensorflow numpy pillow")
    st.write("4. Jalankan aplikasi menggunakan perintah:")
    st.code("streamlit run streamlit_app.py")
