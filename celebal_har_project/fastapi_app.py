import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
from keras.models import load_model

# Load the saved model
vgg_model = Sequential()

pretrained_model = tf.keras.applications.VGG16(
    include_top=False, input_shape=(160, 160, 3), pooling='avg', classes=15, weights='imagenet'
)

for layer in pretrained_model.layers:
    layer.trainable = False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))

# Load the weights from the local file
weights_path = r"C:\Users\SARANSH\celebal_project\model.h5"
vgg_model.load_weights(weights_path)

app = FastAPI()

class InputData(BaseModel):
    input_data: UploadFile

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        prediction = await inference_pipeline(file)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

async def inference_pipeline(image_file: UploadFile) -> str:
    # Read the image file and preprocess it
    contents = await image_file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((160, 160))  # Resize the image if needed
    input_img = np.array(img)  # Convert the image to a numpy array
    
    situation = [
        "sitting", "using_laptop", "hugging", "sleeping", "drinking",
        "clapping", "dancing", "cycling", "calling", "laughing",
        "eating", "fighting", "listening_to_music", "running", "texting"
    ]

    result = vgg_model.predict(np.asarray([input_img]))

    itemindex = np.where(result == np.max(result))
    prediction = situation[itemindex[1][0]]

    return prediction
