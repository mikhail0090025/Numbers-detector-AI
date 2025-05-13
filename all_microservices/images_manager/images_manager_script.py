from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from pathlib import Path
import requests
import tarfile
import shutil

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
    if Path(os.path.join("archive")).is_dir():
        print("Dataset is found")
    else:
        print("Dataset was not found. Downloading...")
        download_dataset()

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

def download_dataset():
    url = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz"
    tar_path = "EnglishImg.tgz"
    extract_path = "archive/numbers/chars74k_png"

    if not Path("archive").is_dir():
        print("Dataset was not found. Downloading...")
        # Скачиваем файл
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tar_path, 'wb') as f:
                f.write(response.raw.read())
            print("Download complete. Extracting...")

            # Распаковываем
            with tarfile.open(tar_path, 'r:gz') as tar:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise ValueError("Attempted Path Traversal in Tar File")
                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tar, extract_path)

            # Удаляем .tgz после распаковки
            os.remove(tar_path)
            print(f"Dataset extracted to {extract_path}")
        else:
            raise Exception(f"Failed to download dataset. Status code: {response.status_code}")
    else:
        print("Dataset is found")