from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np 
import tensorflow as tf
import keras 
from keras.models import load_model

app = Flask(__name__, template_folder='templates', static_folder='static')

model = load_model('models/color_model.h5')
color_labels = ["black", "blue", "brown", "green", "pink", "red", "silver", "white", "yellow"]

@app.route('/color')
def hello_world():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)

        # Cach2
        # binStrImg = request.data
        # arrImg = np.fromstring(binStrImg, np.uint8)
        # image = cv2.imdecode(arrImg, cv2.IMREAD_COLOR)

        image = cv2.resize(image,  (224, 224))
        x = np.expand_dims(image, axis=0)

        my_color = model.predict(x, batch_size=1)
        my_color = color_labels[np.argmax(my_color)]

        return my_color

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)