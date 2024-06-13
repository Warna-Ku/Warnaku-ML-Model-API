import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_cie76
from datetime import datetime
from collections import OrderedDict

app = Flask(__name__)

# Load the model
model_path = "./models/UNet-ResNet34.keras"
model = tf.keras.models.load_model(model_path)

segmentation_labels = OrderedDict({
    'background': [0, 0, 0],
    'lips': [255, 0, 0],
    'eyes': [0, 255, 0],
    'nose': [0, 0, 255],
    'skin': [128, 128, 128],
    'hair': [255, 255, 0],
    'eyebrows': [255, 0, 255],
    'ears': [0, 255, 255],
    'teeth': [255, 255, 255],
    'beard': [255, 192, 192],
    'sunglasses': [0, 128, 128],
})

labels = list(segmentation_labels.keys())
rgb_values = list(segmentation_labels.values())

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def extract_dominant_colors(image, k=3):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return colors

def calculate_rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))

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

def rgb_to_lab(color):
    color = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    color_lab = rgb2lab(color)
    return color_lab[0, 0]

def color_distance(color1, color2):
    return deltaE_cie76(rgb_to_lab(color1), rgb_to_lab(color2))

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    processed_image = preprocess_image(image)

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
    
    #the results
    response = {
        'user_palette': user_palette,
        'created': datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
