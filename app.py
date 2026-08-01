from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import os
import io
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 96

# Load class names
with open(os.path.join(BASE_DIR, 'class_names.json')) as f:
    class_names = json.load(f)

# Load organic remedies knowledge base
with open(os.path.join(BASE_DIR, 'remedies.json'), encoding='utf-8') as f:
    remedies_db = json.load(f)

num_classes = len(class_names)

# Load ONNX model (lightweight, low memory footprint compared to full PyTorch)
onnx_path = os.path.join(BASE_DIR, 'plant_disease_model.onnx')
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0  # HWC, 0-1 range (matches ToTensor())
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, axis=0)  # NCHW
    return arr


def predict_image(image: Image.Image):
    input_tensor = preprocess_image(image)
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    class_name = class_names[pred_idx]
    confidence_pct = round(confidence * 100, 2)

    # Top-3 predictions for transparency
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {'class': class_names[i], 'confidence': round(float(probs[i]) * 100, 2)}
        for i in top3_idx
    ]

    remedy_info = remedies_db.get(class_name, {
        'disease_name': class_name,
        'description': 'No additional information available for this class.',
        'organic_remedies': [],
        'prevention_tips': []
    })

    return {
        'predicted_class': class_name,
        'confidence': confidence_pct,
        'top3': top3,
        'remedy': remedy_info
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = None

        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))
        elif request.is_json:
            data = request.get_json()
            b64_data = data.get('image_base64', '')
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            image_bytes = base64.b64decode(b64_data)
            image = Image.open(io.BytesIO(image_bytes))

        if image is None:
            return jsonify({'error': 'No image provided'}), 400

        result = predict_image(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'classes_loaded': num_classes})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
