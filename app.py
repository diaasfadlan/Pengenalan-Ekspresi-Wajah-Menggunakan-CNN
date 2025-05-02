import streamlit as st
import cv2
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image

# Load model tanpa compile, lalu compile ulang agar sesuai dengan versi Keras/TensorFlow baru
MODEL_PATH = "models/fer2013_mini_XCEPTION.102-0.66.hdf5"
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Daftar label emosi
emotions = ['Marah', 'Disgust', 'Takut', 'Bahagia', 'Sedih', 'Surprise', 'Netral']

# Header aplikasi
st.set_page_config(page_title="Pengenalan Ekspresi Wajah", layout="centered")
st.title("🎭Pengenalan Ekspresi Wajah menggunakan CNN")
st.write("Upload gambar wajah Anda untuk melihat prediksi ekspresinya.")

# Upload file gambar
uploaded_file = st.file_uploader("📤 Upload Gambar (.jpg, .jpeg, .png, .jfif)", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    st.write("Tipe file yang diupload:", uploaded_file.type)

    try:
        # Tampilkan gambar yang diupload
        image = Image.open(uploaded_file).convert('L')  # Convert ke grayscale
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        # Proses gambar untuk prediksi
        image_np = np.array(image)
        resized_img = cv2.resize(image_np, (64, 64))
        input_img = resized_img.reshape(1, 64, 64, 1) / 255.0  # Normalisasi

        # Prediksi
        prediction = model.predict(input_img)
        label = emotions[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Hasil prediksi
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        st.success(f"Ekspresi: **{label}**")
        st.info(f"Akurasi Prediksi: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
