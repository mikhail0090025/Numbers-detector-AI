from PIL import Image, UnidentifiedImageError
import numpy as np
import os

np.random.seed(42)

images_are_loaded = False
images = []
outputs = []
all_folders = [
    os.path.join("archive", "numbers", "chars74k_png", "GoodImg"),
    os.path.join("archive", "numbers", "chars74k_png", "BadImag"),
    os.path.join("archive", "numbers", "chars74k_png", "Fnt"),
    os.path.join("archive", "numbers", "chars74k_png", "Hnd"),
]
def get_images():
    global images, outputs, images_are_loaded
    for folder in all_folders:
        for i in range(10):
            folder_path = os.path.join(folder, f"Sample{i}")
            print(f"folder_path: {folder_path}")
            all_files = os.listdir(folder_path)
            for filename in all_files[:50]:
                try:
                    path = os.path.join(folder_path, filename)
                    print(f"Path: {path}")
                    img = Image.open(path)
                    img = img.convert("RGB")
                    img_resized = img.resize((100, 100), Image.Resampling.LANCZOS)
                    img_array = np.array(img_resized) / 255.0
                    images.append(img_array)
                    outputs.append([0] * 10)
                    outputs[-1][i] = 1
                except UnidentifiedImageError:
                    print(f"Error loading {path}: not an image, skipping")
                    continue

    images = np.array(images)
    outputs = np.array(outputs)
    indexes = np.random.permutation(len(images))
    images = images[indexes]
    outputs = outputs[indexes]
    images_are_loaded = True
    print(f"Images found: {len(images)}")