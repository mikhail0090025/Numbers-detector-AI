import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import io
import time

weights_path = "./checkpoints/my_checkpoint.weights.h5"
input_shape = (100, 100, 3)
start_lr = 0.0001
all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        # keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation="relu"),
        # keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=start_lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    return model

model = create_model()
# Dataset getting
max_attempts = 30
attempt = 0
images = []
outputs = []
images_response = None
outputs_response = None

while attempt < max_attempts:
    try:
        ready_response = requests.get("http://images_manager:5000/is_ready", timeout=10)
        if ready_response.status_code == 200 and ready_response.text == "True":
            images_response = requests.get(f"http://images_manager:5000/images?page={0}&per_page=10000", timeout=10)
            if images_response.status_code == 200:
                outputs_response = requests.get(f"http://images_manager:5000/outputs?page={0}&per_page=10000", timeout=10)
                if outputs_response.status_code == 200:
                    break
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
    attempt += 1
    print(f"Attempt {attempt}/{max_attempts}. Waiting for images_manager...")
    time.sleep(10)  # Увеличил интервал до 10 секунд

if attempt >= max_attempts:
    raise Exception("Failed to connect to images_manager after max attempts")

outputs_response = requests.get(f"http://images_manager:5000/outputs?page={0}&per_page=10000")
images_response = np.load(io.BytesIO(images_response.content))
outputs_response = np.load(io.BytesIO(outputs_response.content))
images.extend(images_response)
outputs.extend(outputs_response)
# After adding images
images = np.array(images)
outputs = np.array(outputs)
print(f"Images count in dataset: {len(images)}")

def go_epochs(epochs_count: int = 10):
    print(f"Images shape: {images.shape}")
    print(f"Outputs shape: {outputs.shape}")
    history = model.fit(images, outputs, epochs=epochs_count, callbacks=[], validation_split=0.2,shuffle=True,batch_size=32)
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])