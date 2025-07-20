import streamlit as st
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

st.title("🖼️ Pengolahan Citra Daun (Preprocessing - Segmentasi - Ekstraksi Ciri)")

uploaded_file = st.file_uploader("Upload gambar daun", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = Image.open("daun.jpg")

# Tampilkan gambar asli
st.subheader("1. Gambar Asli")
st.image(image, use_column_width=True)

# Preprocessing
st.subheader("2. Preprocessing")
img_array = np.array(image)
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
gray = cv2.equalizeHist(gray)
blurred = cv2.medianBlur(gray, 5)
st.image(blurred, caption="Grayscale + Kontras + Filter", use_column_width=True)

# Segmentasi
st.subheader("3. Segmentasi (Threshold)")
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
st.image(binary, caption="Hasil Segmentasi", use_column_width=True)

# Ekstraksi Ciri (GLCM)
st.subheader("4. Ekstraksi Ciri (GLCM)")
glcm = graycomatrix(blurred, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
contrast = graycoprops(glcm, 'contrast')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

st.markdown(f"""
- **Contrast**: `{contrast:.4f}`  
- **Correlation**: `{correlation:.4f}`  
- **Energy**: `{energy:.4f}`  
- **Homogeneity**: `{homogeneity:.4f}`
""")
