# Brain Tumor Classification API

This is a Flask-based API for classifying brain tumor images using a Convolutional Neural Network (CNN). The application accepts MRI scan images, processes them, and returns a prediction indicating whether a tumor is present along with the confidence score.

## Features

- Upload MRI scan images in formats: PNG, JPG, JPEG, GIF.
- Classify images as either "Tumor" or "No Tumor".
- Return the confidence level of the prediction.
- CORS support for frontend integration.

## Technologies

- Flask
- TensorFlow/Keras
- NumPy
- Pillow
- Flask-CORS
- Werkzeug

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/azeem30/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection

2. **Activate Virtual Environment:**

    ```bash
    cd back-end
    .\venv\Scripts\activate

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt

4. **Start the Back end:**

    ```bash
    flask run

5. **Start the Front end:**

    ```bash
    cd front-end
    npm start