
from flask import Flask, render_template, request, redirect, flash, url_for

# machine learning module
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import numpy as np

import os

# directory
os.chdir("/Users/maulanashendi/Desktop/Skripsi/Application 2.0/deployment")

app = Flask(__name__)

# format yang diperbolehkan
ALLOWED_EXTENTION = set(['png', 'jpeg', 'jpg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTION


app.secret_key = 'ST27HL0KP3JHKLP9RQW'

# ==============================================================================================
# web


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload():

    # file agar sesuai dengan request.file
    imagefile = request.files['imagefile']
    if 'imagefile' not in request.files:
        return redirect(request.url)

    # jika tidak ada gambar
    if imagefile.filename == '':
        flash('tidak ada gambar')
        return redirect(request.url)

    # jika gambar sesuai dengan permintaan
    if imagefile and allowed_file(imagefile.filename):
        # upload image
        imagefile = request.files['imagefile']
        image_path = 'static/img/' + imagefile.filename
        image_file = 'img/' + imagefile.filename
        imagefile.save(image_path)
        print()

        # load the image
        model = load_model(os.path.join('static/', 'model 1.0.h5'))
        my_image = load_img(image_path, target_size=(224, 224))

        # preprocess the image
        my_image = img_to_array(my_image)
        my_image = my_image.reshape(
            (1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        my_image = preprocess_input(my_image)

        # make the prediction
        prediction = model.predict(my_image)
        prediction = str(np.round(prediction))

        # logic
        print(prediction)
        if prediction == '[[1. 0. 0. 0.]]':
            result = 'segar'

        elif prediction == '[[0. 1. 0. 0.]]':
            result = 'tidak segar'

        elif prediction == '[[0. 0. 1. 0.]]':
            result = 'cukup segar'
        
        elif prediction == '[[0. 0. 0. 1.]]':
            result = 'sangat tidak segar'

        else:
            result = 'machine learning tidak dapat mengenali gambar', prediction

        return render_template('index.html', result=result, image_path=image_path, image_file=image_file)

    else:
        flash('format gambar salah')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
