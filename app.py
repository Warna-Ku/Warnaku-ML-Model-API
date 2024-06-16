import os
import io
import requests
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_cie76
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define your Google Cloud Storage model URL
MODEL_URL = "https://storage.googleapis.com/warnaku-cs/UNet-ResNet34"

# Download the model file and load it
def load_model_from_url(model_url):
    try:
        # Download the model file from the URL
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            model_bytes = r.content
        
        # Save model bytes to a temporary file
        temp_file = io.BytesIO(model_bytes)
        
        # Load the model using TensorFlow's SavedModel format
        model = tf.keras.models.load_model(temp_file)
        
        return model
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading model: {e}")
    except tf.errors.OpError as e:
        raise RuntimeError(f"Error loading TensorFlow model: {e}")

# Load the TensorFlow model
try:
    model = load_model_from_url(MODEL_URL)
    print('Model loaded successfully.')
except RuntimeError as e:
    print(f"Failed to load model: {e}")

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to extract dominant colors using KMeans clustering
def extract_dominant_colors(image, k=3):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return colors

# Function to calculate root mean square error (RMSE)
def calculate_rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))

# Function to find dominant color in segmented area
def find_dominant_color(segmented_area):
    dominant_colors = extract_dominant_colors(segmented_area)
    best_color = None
    min_rmse = float('inf')

    for color in dominant_colors:
        reconstruction = np.full(segmented_area.shape, color)
        rmse = calculate_rmse(segmented_area, reconstruction)
        if rmse < min_rmse:
            min_rmse = rmse
            best_color = color

    return best_color

# Function to convert RGB to LAB color space
def rgb_to_lab(color):
    color = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    color_lab = rgb2lab(color)
    return color_lab[0, 0]

# Function to calculate color distance in LAB color space
def color_distance(color1, color2):
    return deltaE_cie76(rgb_to_lab(color1), rgb_to_lab(color2))

# Function to determine color palette based on dominant colors
def determine_palette(dominant_colors):
    peach = [255, 229, 180]
    purple = [128, 0, 128]
    default_color = [0, 0, 0]

    lips_color = dominant_colors.get('lips', default_color)
    skin_color = dominant_colors.get('skin', default_color)
    hair_color = dominant_colors.get('hair', default_color)
    eyes_color = dominant_colors.get('eyes', default_color)

    hue = 'warm' if color_distance(lips_color, peach) < color_distance(lips_color, purple) else 'cool'
    skin_saturation = np.linalg.norm(skin_color - np.mean(skin_color))
    saturation_threshold = 20
    saturation = 'bright' if skin_saturation > saturation_threshold else 'muted'
    value_threshold = 127
    mean_brightness = np.mean([np.mean(skin_color), np.mean(hair_color), np.mean(eyes_color)])
    value = 'light' if mean_brightness > value_threshold else 'dark'
    contrast_threshold = 50
    contrast = 'high' if abs(np.mean(hair_color) - np.mean(eyes_color)) > contrast_threshold else 'low'

    metric_vector = [hue == 'warm', saturation == 'bright', value == 'light', contrast == 'high']
    palettes = {
        'spring': [True, True, True, True],
        'summer': [False, True, True, False],
        'autumn': [True, False, False, True],
        'winter': [False, False, False, False]
    }

    min_distance = float('inf')
    best_palette = None
    for season, palette_vector in palettes.items():
        distance = np.sum(np.array(metric_vector) != np.array(palette_vector))
        if distance < min_distance:
            min_distance = distance
            best_palette = season

    return best_palette

# Flask route to predict user palette
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    processed_image = preprocess_image(image)

    try:
        pred_mask = model.predict(processed_image)[0]
        pred_mask = np.argmax(pred_mask, axis=-1)

        segments = {
            'skin': 4,
            'hair': 5,
            'lips': 1,
            'eyes': 2,
        }

        segmented_areas = {}
        for segment, class_idx in segments.items():
            mask = (pred_mask == class_idx)
            segmented_area = processed_image[0][mask]
            if segmented_area.size > 0:
                segmented_areas[segment] = segmented_area

        dominant_colors = {}
        for segment, area in segmented_areas.items():
            if area.size > 0:
                dominant_colors[segment] = find_dominant_color(area)

        user_palette = determine_palette(dominant_colors)
        
        response = {
            'user_palette': user_palette,
            'created': datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
