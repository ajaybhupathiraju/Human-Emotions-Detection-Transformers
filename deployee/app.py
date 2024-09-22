from fastapi import FastAPI, File, UploadFile
from typing import Annotated
import uvicorn
import tensorflow as tf
import cv2
import numpy as np
from transformers import AutoImageProcessor, TFViTModel
import keras
from keras.layers import Dense, Flatten
import warnings

warnings.filterwarnings("ignore")

# define fastapi instance
app = FastAPI()

# predicted classes
class_names = ["angry", "happy", "sad"]


def resize_image(image):
    """This function used to read the uploaded image and resize"""
    try:
        img = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = np.array(img)
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print("Exception raised message :{}".format(e))
        return None


# allow user to upload or input an image.
@app.post('/upload/')
async def _file_upload(
        my_file: Annotated[UploadFile, File(description="An image .jpg file only")]):
    """Allow user to upload or input an image """
    try:
        img = resize_image(await my_file.read())
        if img is not None:
            return predict(img)
    except Exception as e:
        print("Exception raised :{}".format(e))


# ML model prediction
def predict(image):
    """Predict the image using hugging face pretrained model"""
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    inputs = image_processor(image, return_tensors="tf")
    outputs = model(inputs)
    last_hidden_states = outputs.last_hidden_state
    list(last_hidden_states.shape)

    x = last_hidden_states[:, 0, :]

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=32, activation=tf.nn.gelu)(x)
    x = keras.layers.Dense(units=16, activation=tf.nn.gelu)(x)
    x = keras.layers.Dense(units=3, activation="softmax")(x)
    predicted_label = int(tf.math.argmax(x, axis=-1))
    result = class_names[predicted_label]
    print("Predicted class is : {} ".format(result))
    return "Predicted class is : {} ".format(result)


if __name__ == "__main__":
    uvicorn.run(app,host="localhost", port=8001)
