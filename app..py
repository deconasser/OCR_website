from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the pre-trained model
model = load_model('custom_ocr_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file is provided in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Read the image file
        file = request.files['image']
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image to match the model's input requirements
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        # Make a prediction
        predictions = model.predict(image)
        predicted_label = np.argmax(predictions, axis=1)[0]

        return jsonify({'prediction': int(predicted_label)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
