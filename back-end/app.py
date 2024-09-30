from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
CORS(app, resources={"/classify": {"origins": "https://azeem30.github.io/CerebroCheck-Frontend/"}})


logging.basicConfig(level=logging.INFO)

model_path = "./models/brain_tumor_model.h5"
try:
    model = load_model(model_path)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

upload_folder = "./dataset/uploads"
app.config['UPLOAD_FOLDER'] = upload_folder
IMAGE_SIZE = (128, 128)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


def preprocess_image(image_path):
    try:
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None

@app.route('/')
def home():
    return jsonify({"message": "The server is running!"}), 200

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"Error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'Error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'Error': 'Invalid file type. Allowed types: png, jpg, jpeg, gif'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = preprocess_image(file_path)
        if image is None:
            return jsonify({"Error": "Image preprocessing failed"}), 500

        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_names = ['No Tumor', 'Tumor']
        result = class_names[predicted_class]
        confidence = float(np.max(prediction))
        os.remove(file_path)

        return jsonify({'result': result, 'confidence': confidence}), 200
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return jsonify({'Error': 'An error occurred during classification'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
