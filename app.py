import os
import io
import requests
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define your Google Cloud Storage model URL
MODEL_URL = "https://storage.googleapis.com/warnaku-cs/UNet-ResNet34.keras"

# Function to load model from Google Cloud Storage
def load_model_from_url(model_url):
    # Download the model file from the URL
    with requests.get(model_url, stream=True) as r:
        r.raise_for_status()
        model_bytes = r.content
    
    # Save model bytes to a temporary file
    temp_file = io.BytesIO(model_bytes)
    
    # Load the model using Keras
    model = tf.keras.models.load_model(temp_file)
    
    return model

# Load the model
model = load_model_from_url(MODEL_URL)
print('Model loaded successfully.')

# Save the model to .h5 file (optional, for reference)
model.save('model.h5')
print('Model saved to model.h5')

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Flask route to predict user palette
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    processed_image = preprocess_image(image)

    # Predict using the loaded model
    prediction = model.predict(processed_image)
    
    # Example response
    response = {
        'prediction': prediction.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
