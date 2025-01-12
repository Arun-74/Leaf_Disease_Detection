from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    return input_arr

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and make a prediction
            input_arr = preprocess_image(file_path)
            predictions = model.predict(input_arr)
            result_index = np.argmax(predictions)

            # Map the index to the class name
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                           'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 
                           'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 
                           'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 
                           'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                           'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
                           'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                           'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                           'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                           'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                           'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                           'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                           'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

            model_prediction = class_names[result_index]

            # Remove the uploaded file after prediction
            os.remove(file_path)

            return render_template('result.html', prediction=model_prediction)
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
