from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('custom_ocr_model.keras')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    image = Image.open(file.stream).convert('L')
    image = image.resize((28,28))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    predict = model.predict(image)
    res = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index = np.argmax(predict)
    return render_template("index.html", predict=res[index])

if __name__ == '__main__':
    app.run()
