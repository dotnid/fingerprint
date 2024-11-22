from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import pandas as pd
from flask_ngrok import run_with_ngrok

# Replace YOUR_AUTHTOKEN with your actual ngrok authtoken
os.system('ngrok authtoken 2bMeQV0gd5eBwdqiqX2dNva7zqO_4DbQpt6SkdXohy6EUdkBT')

# Inisialisasi aplikasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Folder untuk menyimpan file upload
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
run_with_ngrok(app)  # Add this line
# Buat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class ImagePreprocessor:
    def __init__(self, target_size=128, crop_size=128, normalization_value=64.0):
        self.target_size = target_size
        self.crop_size = crop_size
        self.normalization_value = normalization_value

    def get_center_crop(self, img):
        height, width = img.shape
        startx = width // 2 - (self.crop_size // 2)
        starty = height // 2 - (self.crop_size // 2)
        return img[starty:starty + self.crop_size, startx:startx + self.crop_size]

    def preprocess_image(self, img):
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((self.target_size, self.target_size))  # Resizing image
        img_array = np.array(img)  # Convert image to array
        cropped_img_array = self.get_center_crop(img_array)  # Cropping image
        normalized_img = cropped_img_array / self.normalization_value  # Normalize
        return normalized_img.reshape((self.crop_size, self.crop_size, 1))

# Membuat instance dari ImagePreprocessor
preprocessor = ImagePreprocessor(target_size=128, crop_size=128)

# Muat model dan personality dataframe
model = load_model('model/model.h5')  # Ganti dengan path ke file model Anda
personality_df = pd.read_csv('model/personality.csv')  # Pastikan file personality.csv tersedia

# Buat dictionary untuk mapping id ke target dan deskripsi
id_to_target = pd.Series(personality_df.target.values, index=personality_df.id).to_dict()
id_to_description = pd.Series(personality_df.description.values, index=personality_df.id).to_dict()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('userInput.html')

@app.route('/karakter')
def karakter():
    return render_template('karakter.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Simpan file ke folder uploads
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            img = Image.open(file_path)
            
            # Preprocess gambar menggunakan ImagePreprocessor
            processed_img = preprocessor.preprocess_image(img)
            processed_img = np.expand_dims(processed_img, axis=0)  # Menambahkan dimensi batch
            predictions = model.predict(processed_img)
            
            # Misalnya, jika model menggunakan softmax untuk klasifikasi
            predicted_class_index = np.argmax(predictions[0])
            predicted_label = id_to_target.get(predicted_class_index, 'Unknown')
            predicted_description = id_to_description.get(predicted_class_index, 'No description available')

            # Mengirimkan hasil prediksi dalam response JSON
            return jsonify({
                'predicted_label': predicted_label,
                'description': predicted_description,
                'image_url': f'/uploads/{file.filename}'
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File upload failed'}), 400

if __name__ == '__main__':
    app.run()
