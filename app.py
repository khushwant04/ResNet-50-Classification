from flask import Flask, request, jsonify, render_template
from src.models import ResNet50
from src.dataset import Animals
from src.transforms import make_transforms
from flask_cors import CORS
from PIL import Image
import io
import torch


animals = Animals(root='src/data/animal_data')
class_labels = animals.get_class_to_idx()

app = Flask(__name__)
CORS(app)

# Load the model
model = ResNet50(num_classes=15)  # Replace YourModelClass with your actual PyTorch model class
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define image transformation
transform = make_transforms()

# Class labels (replace with your actual class labels)
class_labels = list(class_labels.keys())

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
        img = Image.open(io.BytesIO(file.read()))
        result = classify_image(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
