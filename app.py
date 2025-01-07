import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

def load_image(filePath):
    image = tf.io.read_file(filePath)

    image = tf.image.decode_image(image, channels = 1)

    image = tf.image.resize(image, (224, 224))

    image = image/255.0

    image = tf.reshape(image, [-1])

    image = image.numpy()

    image = image.reshape(1,50176)

    return image

# Load the trained model
MODEL_PATH = 'brain_tumor_model.h5'
model = load_model(MODEL_PATH)

# Define the input image size
IMG_SIZE = (224, 224)  # Replace with the size your model expects

# Home page route
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML file for the UI

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)  # Create 'uploads' folder if not exists
        file.save(file_path)

        # Preprocess the image
        img = load_image(file_path)
        

        # Make prediction
        prediction = model.predict(img)
        probability = tf.nn.sigmoid(prediction[0][0]).numpy()
        
        if probability > 0.5:
            tumor_present = True
            confidence = probability

        else:
            tumor_present = False
            confidence = 1 - probability


        # Prepare result
        result = "Tumor Detected" if tumor_present else "No Tumor Detected"
        return render_template('result.html', result=result, image_path=file_path, confidence=confidence*100)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)