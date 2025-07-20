import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess(img):
    # Preprocessing: grayscale & Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return blur

def segment(img):
    # Segmentasi: Otsu thresholding
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask

def extract_features(img):
    # Ekstraksi ciri: Haralick (GLCM)
    glcm = graycomatrix(img, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    return f"Contrast: {contrast:.2f}, Homogeneity: {homogeneity:.2f}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    preprocessed = preprocess(img)
    segmented = segment(preprocessed)
    features = extract_features(preprocessed)

    # Save processed images
    pre_path = os.path.join(UPLOAD_FOLDER, 'pre_'+file.filename)
    seg_path = os.path.join(UPLOAD_FOLDER, 'seg_'+file.filename)
    cv2.imwrite(pre_path, preprocessed)
    cv2.imwrite(seg_path, segmented)

    result = {
        'original': filepath,
        'preprocessed': pre_path,
        'segmented': seg_path,
        'features': features
    }
    return render_template('index.html', result=result)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)