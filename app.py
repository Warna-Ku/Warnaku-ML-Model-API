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
from collections import OrderedDict

app = Flask(__name__)

# Define your Google Cloud Storage model URL
MODEL_URL = "https://storage.googleapis.com/warnaku-cs/UNet-ResNet34.keras"

# Download the model file and load it
def load_model_from_url(model_url):
    try:
        # Download the model file from the URL
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            model_bytes = r.content
        
        # Save model bytes to a temporary file with .keras extension
        temp_file_path = "/tmp/model.keras"
        with open(temp_file_path, "wb") as f:
            f.write(model_bytes)
        
        # Load the model using TensorFlow's load_model function
        model = tf.keras.models.load_model(temp_file_path)
        
        return model
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading model: {e}")
    except tf.errors.OpError as e:
        raise RuntimeError(f"Error loading TensorFlow model: {e}")

# Load the TensorFlow model and store in Flask's application context
@app.before_first_request
def load_model():
    global model
    model = load_model_from_url(MODEL_URL)
    print('Model loaded successfully.')

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
    
    # Construct response with the desired key order
    response = OrderedDict()
    response['user_palette'] = user_palette
    response['created'] = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')

    # Mapping palette to GCS URLs
    bucket_url = "https://storage.googleapis.com/warnaku-cs"
    palette_info = {
        "autumn": {
            'PalImg': f"{bucket_url}/assets/AutumnPal.png",
            'PalDesc': "Palet musim gugur menghadirkan kehangatan dan kedalaman dengan warna-warna seperti coklat tua, oranye hangat, dan hijau zaitun. Warna-warna ini sempurna untuk pernikahan yang memancarkan keintiman dan kenyamanan dalam suasana alami dan hangat.",
            'Colors': [
                {"Desc": "Kombu Green yang tenang ini cocok untuk gaun pengantin yang elegan di tengah suasana alami yang hangat.", "Img": f"{bucket_url}/Autumn_Cloth1_KombuGreen.jpg", "Code": "#354230", "Name": "Kombu Green"},
                {"Desc": "Old Moss Green menciptakan nuansa alami dan hangat, ideal untuk setelan pengantin pria yang kuat dan intim.", "Img": f"{bucket_url}/Autumn_Cloth2_OldMossGreen.jpg", "Code": "#867E36", "Name": "Old Moss Green"},
                {"Desc": "Outer Space memberikan sentuhan modern pada dekorasi pernikahan, menambah kesan tegas dan maskulin.", "Img": f"{bucket_url}/Autumn_Cloth3_OuterSpace.jpg", "Code": "#414A4C", "Name": "Outer Space"},
                {"Desc": "Brownish Black menambahkan elemen misterius dan dramatis, cocok untuk gaun pesta yang mempesona.", "Img": f"{bucket_url}/Autumn_Cloth4_BrownishBlack.jpg", "Code": "#665D1E", "Name": "Brownish Black"},
                {"Desc": "Metallic Bronze mengekspresikan semangat dan kegembiraan, sempurna untuk aksen dekorasi yang riang.", "Img": f"{bucket_url}/Autumn_Cloth5_MetallicBronze.jpg", "Code": "#B0855B", "Name": "Metallic Bronze"}
            ]
        },
        "spring": {
            'PalImg': f"{bucket_url}/assets/SpringPal.png",
            'PalDesc': "Palet musim semi membawa kebahagiaan dan keceriaan dengan warna-warna cerah seperti hijau muda, kuning cerah, dan merah muda pastel. Palet ini cocok untuk pernikahan yang menyenangkan dan penuh semangat di tengah suasana ceria dan hidup.",
            'Colors': [
                {"Desc": "Bright Green ini memberikan nuansa segar dan cerah, sempurna untuk dekorasi bunga di pernikahan musim semi.", "Img": f"{bucket_url}/Spring_Cloth1_BrightGreen.jpg", "Code": "#66FF66", "Name": "Bright Green"},
                {"Desc": "Vivid Yellow menambah sentuhan ceria pada gaun pengantin wanita, membawa keceriaan dan kebahagiaan.", "Img": f"{bucket_url}/Spring_Cloth2_VividYellow.jpg", "Code": "#FFFF66", "Name": "Vivid Yellow"},
                {"Desc": "Light Pink menciptakan suasana lembut dan romantis, ideal untuk aksen dekorasi yang mempesona.", "Img": f"{bucket_url}/Spring_Cloth3_LightPink.jpg", "Code": "#FFB6C1", "Name": "Light Pink"},
                {"Desc": "Baby Blue memberikan nuansa tenang dan damai, cocok untuk gaun pengiring pengantin yang menenangkan.", "Img": f"{bucket_url}/Spring_Cloth4_BabyBlue.jpg", "Code": "#89CFF0", "Name": "Baby Blue"},
                {"Desc": "Lavender memberikan sentuhan elegan dan anggun pada pesta pernikahan, menambah kesan mewah.", "Img": f"{bucket_url}/Spring_Cloth5_Lavender.jpg", "Code": "#E6E6FA", "Name": "Lavender"}
            ]
        },
        "summer": {
            'PalImg': f"{bucket_url}/assets/SummerPal.png",
            'PalDesc': "Palet musim panas menghadirkan keceriaan dan kesegaran dengan warna-warna terang seperti biru laut, hijau mint, dan putih cerah. Palet ini cocok untuk pernikahan yang penuh semangat dan energi di tengah suasana yang terang dan hangat.",
            'Colors': [
                {"Desc": "Turquoise ini memberikan kesan segar dan dinamis, ideal untuk dekorasi pantai di pernikahan musim panas.", "Img": f"{bucket_url}/Summer_Cloth1_Turquoise.jpg", "Code": "#30D5C8", "Name": "Turquoise"},
                {"Desc": "Coral Pink menambah sentuhan ceria dan feminin, cocok untuk gaun pengiring pengantin yang menyenangkan.", "Img": f"{bucket_url}/Summer_Cloth2_CoralPink.jpg", "Code": "#F88379", "Name": "Coral Pink"},
                {"Desc": "Sunny Yellow menciptakan suasana hangat dan penuh semangat, ideal untuk aksen dekorasi yang ceria.", "Img": f"{bucket_url}/Summer_Cloth3_SunnyYellow.jpg", "Code": "#FFD700", "Name": "Sunny Yellow"},
                {"Desc": "Ocean Blue memberikan nuansa tenang dan menenangkan, sempurna untuk dekorasi meja di pesta pernikahan.", "Img": f"{bucket_url}/Summer_Cloth4_OceanBlue.jpg", "Code": "#0077BE", "Name": "Ocean Blue"},
                {"Desc": "Mint Green menambahkan kesan segar dan alami, cocok untuk buket bunga yang mempesona.", "Img": f"{bucket_url}/Summer_Cloth5_MintGreen.jpg", "Code": "#98FF98", "Name": "Mint Green"}
            ]
        },
        "winter": {
            'PalImg': f"{bucket_url}/assets/WinterPal.png",
            'PalDesc': "Palet musim dingin menghadirkan keanggunan dan kemewahan dengan warna-warna seperti biru tua, putih salju, dan merah marun. Palet ini cocok untuk pernikahan yang memancarkan keindahan dan kemegahan dalam suasana yang elegan dan dingin.",
            'Colors': [
                {"Desc": "Royal Blue memberikan kesan anggun dan mewah, ideal untuk gaun pengantin yang megah.", "Img": f"{bucket_url}/Winter_Cloth1_RoyalBlue.jpg", "Code": "#4169E1", "Name": "Royal Blue"},
                {"Desc": "Burgundy menambah sentuhan hangat dan mewah, cocok untuk dekorasi meja yang elegan.", "Img": f"{bucket_url}/Winter_Cloth2_Burgundy.jpg", "Code": "#800020", "Name": "Burgundy"},
                {"Desc": "Snow White menciptakan suasana dingin dan anggun, sempurna untuk aksen dekorasi yang mempesona.", "Img": f"{bucket_url}/Winter_Cloth3_SnowWhite.jpg", "Code": "#FFFAFA", "Name": "Snow White"},
                {"Desc": "Slate Gray memberikan nuansa modern dan kuat, cocok untuk setelan pengantin pria yang tegas.", "Img": f"{bucket_url}/Winter_Cloth4_SlateGray.jpg", "Code": "#708090", "Name": "Slate Gray"},
                {"Desc": "Crimson Red menambahkan elemen dramatis dan romantis, ideal untuk gaun pesta yang memukau.", "Img": f"{bucket_url}/Winter_Cloth5_CrimsonRed.jpg", "Code": "#DC143C", "Name": "Crimson Red"}
            ]
        }
    }

    response['PalDesc'] = palette_info[user_palette]['PalDesc']
    response['PalImg'] = palette_info[user_palette]['PalImg']
    response['Colors'] = palette_info[user_palette]['Colors']

    return jsonify(response)
if __name__ == '__main__':
    app.run(port=5000, debug=True)
