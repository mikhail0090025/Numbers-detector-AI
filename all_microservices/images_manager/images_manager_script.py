from PIL import Image, UnidentifiedImageError
import numpy as np
import os

images_are_loaded = False
images = []
outputs = []
for i in range(10):
    folder_path = os.path.join("archive", "numbers", "chars74k_png", "GoodImg", f"Sample{i}")
    print(f"folder_path: {folder_path}")
    all_files = os.listdir(folder_path)
    for filename in all_files[:10]:
        try:
            path = os.path.join(folder_path, filename)
            print(f"Path: {path}")
            img = Image.open(path)
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
images_are_loaded = True
print(f"Images found: {len(images)}")