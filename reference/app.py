#!/usr/bin/env python3

from flask import Flask, request, jsonify
from urllib.request import urlopen
from PIL import Image

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten
from keras.applications.resnet50 import ResNet50
from keras.models import load_model


batch_size = 32

vgg_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

model_aug = Sequential()
model_aug.add(vgg_model)

top_model = Sequential()
top_model.add(Flatten(input_shape=(9, 9, 512)))
# model_aug.add(Dropout(0.3))
top_model.add(Dense(100, activation='relu'))

top_model.add(Dense(7, activation='sigmoid'))

model_aug.add(top_model)

for layer in model_aug.layers[0].layers[:17]:
    layer.trainable = False

model_aug.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-6), metrics=['accuracy'])
#model_aug.save("model_final.h5")


#model_aug = load_model("model_final.h5")
model_aug._make_predict_function()


app = Flask(__name__)

@app.route("/predict", methods = ['POST'])
def home():
    data = request.get_json()
    im = Image.open(urlopen(data['img_url'])).convert("RGB")

    im = im.resize((300, 300), Image.ANTIALIAS)
    im_test = im.resize((4, 4), Image.ANTIALIAS)


    im = np.array(im).reshape(1, 300, 300, 3)

    res = predict(im)

    cat_dict = {0:'ADULT', 1:'COUNTERFEIT', 2:'LEGIT', 3:'PHARMA', 4:'SMOKE', 5:'TMS', 6:'WEAPON'}



    ret = {
        'category': cat_dict[res[0][0].tolist()],
        'probabilities':  list(zip(cat_dict.values() , res[1][0].tolist())),
    }


    return jsonify(ret)

@app.route("/test") # default: GET
def test():
    im_url = "https://images-na.ssl-images-amazon.com/images/I/31xccEdro8L.jpg"
    im = Image.open(urlopen(im_url)).convert("RGB")

    ret = {
        'im': np.array(im).tolist()
    }

    return jsonify(ret)



def predict(im):

    # shape(im) = (m, 300, 300, 3)
    return model_aug.predict_classes(im), model_aug.predict(im)

if __name__ == "__main__":
    app.run(host='10.8.1.203', debug=True, port = 4999)