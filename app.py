import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Classification", layout="centered")

st.title("Klasifikasi Gambar Buah")

uploaded_model = st.file_uploader("Upload model (.h5)", type=["h5"])
uploaded_image = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])

img_size = (277, 277)
class_names_text = st.text_input(
    "Nama kelas (pisahkan dengan koma)",
    value="anggur,buah naga"
)

if uploaded_model and uploaded_image:
    class_names = [x.strip() for x in class_names_text.split(",")]

    with open("temp_model.h5", "wb") as f:
        f.write(uploaded_model.read())

    model = load_model("temp_model.h5", compile=False)

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
