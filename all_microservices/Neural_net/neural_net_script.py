import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import io
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

weights_path = "./my_checkpoint.weights.h5"
input_shape = (40, 40, 3)
start_lr = 1e-4
all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Dropout(0.1),
        keras.layers.Conv2D(64, (3,3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Dropout(0.1),
        keras.layers.Conv2D(128, (3,3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2048, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=start_lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()

    try:
        model.load_weights(weights_path)
        print("Saved model was loaded")
    except FileNotFoundError as e:
        print(f"No model was saved in path {weights_path}. New will be created")
    except Exception as e:
        print(f"Weights were not loaded ({e}). New will be created")

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
    time.sleep(10)

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
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.9, 1.1],
        horizontal_flip=False,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow(
        images,
        outputs,
        batch_size=64,
        subset='training',
        shuffle=True
    )
    val_generator = val_datagen.flow(
        images,
        outputs,
        batch_size=64,
        subset='validation',
        shuffle=False
    )
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy', factor=0.5, patience=3, min_lr=1e-6
    )
    SaveCheckpoint = tf.keras.callbacks.ModelCheckpoint(weights_path,
                                     monitor='accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='auto',
                                     save_freq='epoch')
    history = model.fit(
        train_generator,
        epochs=epochs_count,
        validation_data=val_generator,
        callbacks=[SaveCheckpoint, lr_scheduler]
    )
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])