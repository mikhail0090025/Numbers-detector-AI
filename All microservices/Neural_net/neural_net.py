import tensorflow
from tensorflow import keras
import numpy as np

weights_path = "./checkpoints/my_checkpoint.weights.h5"
input_shape = (100, 100, 3)

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation="relu")
    ])