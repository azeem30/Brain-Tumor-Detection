from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)
model_path = "./models/brain_tumor_model.h5"
model = load_model(model_path)
upload_folder = "./dataset/uploads"
app.config['UPLOAD_FOLDER'] = upload_folder

IMAGE_SIZE = (128, 128)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def preprocess_image(image_path):
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"Error": "No File Part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'Error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'Error': 'Invalid file type. Allowed types: png, jpg, jpeg, gif'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        os.remove(file_path)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_names = ['No Tumor', 'Tumor']
        result = class_names[predicted_class]
        return jsonify({'result': result, 'confidence': float(np.max(prediction))})
    return jsonify({'Error': 'Invalid file upload'}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)