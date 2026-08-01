from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import io
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 96
device = torch.device('cpu')

# Load class names
with open(os.path.join(BASE_DIR, 'class_names.json')) as f:
    class_names = json.load(f)

# Load organic remedies knowledge base
with open(os.path.join(BASE_DIR, 'remedies.json'), encoding='utf-8') as f:
    remedies_db = json.load(f)

num_classes = len(class_names)


class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = PlantCNN(num_classes)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'plant_disease_model.pth'), map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def predict_image(image: Image.Image):
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)

    class_name = class_names[pred_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)

    # Top-3 predictions for transparency
    top3_prob, top3_idx = torch.topk(probs, k=min(3, num_classes))
    top3 = [
        {'class': class_names[i.item()], 'confidence': round(p.item() * 100, 2)}
        for p, i in zip(top3_prob, top3_idx)
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

        # Case 1: image uploaded as file (from <input type="file">)
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))

        # Case 2: image sent as base64 string (from live camera capture)
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
