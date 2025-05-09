from PIL import Image
import numpy as np
import os

images = []
for i in range(0, 10):
    all_files = os.listdir(f"archive\\numbers\chars74k_png\GoodImg\Sample{i}")
    for j in range(0, 100):
        if j >= len(all_files):
            print("All files from this folder are processed.")
            break
        path = f"archive\\numbers\chars74k_png\GoodImg\Sample{i}\{all_files[j]}"
        print(f"Path: {path}")
        img = Image.open(path)
        img_resized = img.resize((100, 100), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)  # Shape: (100, 100, 3)
        img_array = img_array / 255.0  # Для [0, 1]
        images.append(img_array)

print(f"Images found: {len(images)}")
'''
img_gray = img_resized.convert("L")  # Конвертируем в grayscale
img_array = np.array(img_gray)  # Shape: (100, 100)
img_tensor = np.expand_dims(img_array, axis=0)  # Для PyTorch: (1, 100, 100)
img_tensor = np.expand_dims(img_tensor, axis=0)  # Если нужно (1, 1, 100, 100)
'''