import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from keras.models import load_model
from PIL import Image

# Load the saved model
vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))

# Load the weights from the local file
weights_path = r"C:\Users\SARANSH\celebal_project\model.h5"
vgg_model.load_weights(weights_path)


@require_POST
def predict(request):
    # Get the input data from the POST request
    input_data = request.FILES.get("input_data")
    image = Image.open(input_data)  

    situation=["sitting","using_laptop","hugging","sleeping","drinking",
           "clapping","dancing","cycling","calling","laughing"
          ,"eating","fighting","listening_to_music","running","texting"]

    input_img = np.asarray(image.resize((160,160)))
    result = vgg_model.predict(np.asarray([input_img]))

    itemindex = np.where(result==np.max(result))
    prediction = situation[itemindex[1][0]]

    # Return the predictions as a JSON response
    return JsonResponse({"prediction": prediction})