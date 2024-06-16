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
    
    response = {
        'user_palette': user_palette,
        'created': datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

    # Add detailed palette descriptions and images based on the user's palette
    if user_palette == "autumn":
        response.update({
            'PalDesc': "Palet musim gugur menghadirkan kehangatan dan kedalaman dengan warna-warna seperti coklat tua, oranye hangat, dan hijau zaitun. Warna-warna ini sempurna untuk pernikahan yang memancarkan keintiman dan kenyamanan dalam suasana alami dan hangat.",
            'Colors': [
                {"Desc": "Kombu Green yang tenang ini cocok untuk gaun pengantin yang elegan di tengah suasana alami yang hangat.", "Img": "Autumn_Cloth1_KombuGreen.jpg", "Code": "#354230", "Name": "Kombu Green"},
                {"Desc": "Old Moss Green menciptakan nuansa alami dan hangat, ideal untuk setelan pengantin pria yang kuat dan intim.", "Img": "Autumn_Cloth2_OldMossGreen.jpg", "Code": "#867E36", "Name": "Old Moss Green"},
                {"Desc": "Outer Space memberikan sentuhan modern pada dekorasi pernikahan, menambah kesan tegas dan maskulin.", "Img": "Autumn_Cloth3_OuterSpace.jpg", "Code": "#414A4C", "Name": "Outer Space"},
                {"Desc": "Brownish Black menambahkan elemen misterius dan dramatis, cocok untuk gaun pesta yang mempesona.", "Img": "Autumn_Cloth4_BrownishBlack.jpg", "Code": "#665D1E", "Name": "Brownish Black"},
                {"Desc": "Metallic Bronze mengekspresikan semangat dan kegembiraan, sempurna untuk aksen dekorasi yang riang.", "Img": "Autumn_Cloth5_MetallicBronze.jpg", "Code": "#B08554", "Name": "Metallic Bronze"}
            ],
            'PalImg': "AutumnPal.png"
        })
    elif user_palette == "spring":
        response.update({
            'PalDesc': "Palet musim semi membawa palet cerah dan segar, dengan warna biru cerah, hijau muda, dan merah muda. Ini mencerminkan kebangkitan dan harapan baru, cocok untuk pernikahan yang penuh keceriaan di taman berbunga.",
            'Colors': [
                {"Desc": "Blue Purple melambangkan kedamaian dan kebangkitan, cocok untuk gaun pengantin yang segar di musim semi.", "Img": "Spring_Cloth1_BluePurple.jpg", "Code": "#745AA9", "Name": "Blue Purple"},
                {"Desc": "Cyan Blue memberikan sentuhan ringan dan ceria, ideal untuk dasi atau aksesori pengantin pria.", "Img": "Spring_Cloth2_CyanBlue.jpg", "Code": "#00B7EB", "Name": "Cyan Blue"},
                {"Desc": "Green mengingatkan pada dedaunan segar, cocok untuk dekorasi bunga yang melambangkan kehidupan baru.", "Img": "Spring_Cloth3_Green.jpg", "Code": "#00FF00", "Name": "Green"},
                {"Desc": "Crayola Orange Red membawa energi dan keceriaan, pas untuk gaun pesta yang mempesona.", "Img": "Spring_Cloth4_CrayolaOrangeRed.jpg", "Code": "#FF4433", "Name": "Crayola Orange Red"},
                {"Desc": "Princeton Orange mencerminkan keceriaan dan semangat, cocok untuk aksen dekoratif yang menciptakan suasana riang.", "Img": "Spring_Cloth5_PrincetonOrange.jpg", "Code": "#FF7036", "Name": "Princeton Orange"}
            ],
            'PalImg': "SpringPal.png"
        })
    elif user_palette == "summer":
        response.update({
            'PalDesc': "Palet musim panas menampilkan warna-warna yang cerah dan ceria seperti biru laut, ungu muda, dan merah jambu. Ideal untuk pernikahan pantai atau luar ruangan yang menyenangkan dan riang, mencerminkan kebebasan dan keceriaan musim panas.",
            'Colors': [
                {"Desc": "Crystal Light Blue menghadirkan ketenangan dan kebebasan, sempurna untuk gaun pengantin yang ringan dan mengalir.", "Img": "Summer_Cloth1_CrystalLightBlue.jpg", "Code": "#ACE6FB", "Name": "Crystal Light Blue"},
                {"Desc": "Cool Grey menambahkan sentuhan elegan dan romantis, ideal untuk gaun pengiring pengantin yang manis.", "Img": "Summer_Cloth2_CoolGrey.jpg", "Code": "#8C92AC", "Name": "Cool Grey"},
                {"Desc": "Jungle Green memberikan kesan segar dan eksotis, cocok untuk aksen dekorasi pantai atau tropis.", "Img": "Summer_Cloth3_JungleGreen.jpg", "Code": "#29AB87", "Name": "Jungle Green"},
                {"Desc": "Brick Red memancarkan semangat dan gairah, pas untuk gaun pesta yang memukau dan berani.", "Img": "Summer_Cloth4_BrickRed.jpg", "Code": "#CB4154", "Name": "Brick Red"},
                {"Desc": "Metallic Pink menambahkan kelembutan dan keanggunan, sempurna untuk detail dekorasi bunga atau aksesoris pengantin.", "Img": "Summer_Cloth5_MetallicPink.jpg", "Code": "#FF8AAB", "Name": "Metallic Pink"}
            ],
            'PalImg': "SummerPal.png"
        })
    elif user_palette == "winter":
        response.update({
            'PalDesc': "Musim dingin membawa palet yang tajam dan dramatis dengan warna hitam, putih, dan merah tua. Cocok untuk pernikahan dalam ruangan yang elegan dan mewah, menciptakan suasana berkelas dan memikat yang penuh dengan kemewahan.",
            'Colors': [
                {"Desc": "Lemon Yellow membawa kecerahan dan kegembiraan, cocok untuk aksen dekorasi yang menyegarkan suasana pernikahan musim dingin.", "Img": "Winter_Cloth1_LemonYellow.jpg", "Code": "#FFF700", "Name": "Lemon Yellow"},
                {"Desc": "Black Chocolate menambahkan sentuhan elegan dan dramatis, sempurna untuk setelan pengantin pria yang klasik dan berkelas.", "Img": "Winter_Cloth2_BlackChocolate.jpg", "Code": "#1B1811", "Name": "Black Chocolate"},
                {"Desc": "Floral White menciptakan suasana kemurnian dan kesucian, cocok untuk gaun pengantin yang abadi dan indah.", "Img": "Winter_Cloth3_FloralWhite.jpg", "Code": "#FFFAF0", "Name": "Floral White"},
                {"Desc": "Dark Crimson menambahkan nuansa hangat dan romantis, pas untuk gaun pengiring pengantin yang memukau.", "Img": "Winter_Cloth4_DarkCrimson.jpg", "Code": "#8C0000", "Name": "Dark Crimson"},
                {"Desc": "Metallic Green menciptakan suasana alami dan tenang, sempurna untuk dekorasi bunga atau aksesoris yang elegan.", "Img": "Winter_Cloth5_MetallicGreen.jpg", "Code": "#0A694F", "Name": "Metallic Green"}
            ],
            'PalImg': "WinterPal.png"
        })
    else:
        response.update({'PalDesc': "Invalid palette"})

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
