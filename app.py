
import streamlit as st
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

st.set_page_config(layout="wide")
st.title("🌿 Aplikasi Pengolahan Citra")

st.sidebar.header("📂 Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "png"])


if uploaded_file is not None:
    # Load gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (300, 300))

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Segmentasi
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ekstraksi ciri
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea) if contours else None
    luas = cv2.contourArea(cnt) if cnt is not None else 0
    keliling = cv2.arcLength(cnt, True) if cnt is not None else 0

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    kontras = graycoprops(glcm, 'contrast')[0, 0]
    homogenitas = graycoprops(glcm, 'homogeneity')[0, 0]

    mask = thresh == 255
    rata_r = np.mean(img[:, :, 2][mask])
    rata_g = np.mean(img[:, :, 1][mask])
    rata_b = np.mean(img[:, :, 0][mask])

    # Tampilkan gambar hasil
    st.subheader("🖼️ Hasil Pengolahan Gambar")
    col1, col2, col3, col4 = st.columns(4)
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Asli", use_container_width=True)
    col2.image(gray, caption="Grayscale", use_container_width=True, channels="GRAY")
    col3.image(blur, caption="Blur", use_container_width=True, channels="GRAY")
    col4.image(thresh, caption="Segmentasi", use_container_width=True, channels="GRAY")
    

    # Tampilkan fitur
    st.subheader("📊 Ekstraksi Fitur:")
    st.markdown(f"""
    - **Luas:** {luas:.2f} piksel  
    - **Keliling:** {keliling:.2f} piksel  
    - **Tekstur Kontras:** {kontras:.4f}  
    - **Tekstur Homogenitas:** {homogenitas:.4f}  
    - **Rata-rata Warna R:** {rata_r:.2f}  
    - **Rata-rata Warna G:** {rata_g:.2f}  
    - **Rata-rata Warna B:** {rata_b:.2f}  
    """)
else:
    st.info("📤 Silakan unggah gambar untuk mulai.")
