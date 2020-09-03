from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/food_vgg.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()   

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        num = request.form.get('num')
        items = int(num)
        chapati = 85
        dosa = 104
        idly = 39

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = model_predict(file_path, model)

            
        prediction = np.round(prediction)
        if prediction[0][0] == 1:
          total = chapati * items
          result = 'Total Calories in your meal(chapati) is '+ str(total) + '.'
        elif prediction[0][1] == 1:
          total = dosa * items
          result = 'Total Calories in your meal(dosa) is '+ str(total) + '.'
        elif prediction[0][2] == 1:
          total = idly * items  
          result = 'Total Calories in your meal(idly) is '+ str(total) +'.'



        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

