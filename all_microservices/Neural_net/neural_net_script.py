import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import io

weights_path = "./checkpoints/my_checkpoint.weights.h5"
input_shape = (100, 100, 3)
all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation="relu"),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    return model

model = create_model()
# Dataset getting
images = []
outputs = []
images_response = requests.get(f"http://images_manager:5000/images?page={0}&per_page=100")
outputs_response = requests.get(f"http://images_manager:5000/outputs?page={0}&per_page=100")
images_response = np.load(io.BytesIO(images_response.content))
outputs_response = np.load(io.BytesIO(outputs_response.content))
images.extend(images_response)
outputs.extend(outputs_response)
# After adding images
images = np.array(images)
outputs = np.array(outputs)

def go_epochs(epochs_count: int = 10):
    print(f"Images shape: {images.shape}")
    print(f"Outputs shape: {outputs.shape}")
    history = model.fit(images, outputs, epochs=epochs_count, callbacks=[], validation_split=0.2,shuffle=True,batch_size=32)
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])