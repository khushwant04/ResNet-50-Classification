from flask import Flask, request, jsonify, render_template
from src.models import ResNet50
from src.transforms import make_transforms
from flask_cors import CORS
from PIL import Image
import io
import os
import torch


# Class labels in training index order (sorted folder names of src/data/animal_data,
# as printed by Animals.get_class_to_idx() in main.ipynb). Hardcoded so the API
# does not need the training dataset on disk to start.
class_labels = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
    'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra',
]

app = Flask(__name__)
CORS(app)

# Load the model (checkpoint was saved from a CUDA device, so map it to CPU)
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pth')
model = ResNet50(num_classes=len(class_labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Define image transformation
transform = make_transforms()

# Function to perform image classification
def classify_image(image):
    image = transform(image).unsqueeze(0)  # Preprocess the image
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_index = predicted.item()
    return class_labels[class_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        result = classify_image(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', '0') == '1',
    )
