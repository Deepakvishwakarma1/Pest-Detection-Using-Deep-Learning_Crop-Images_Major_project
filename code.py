import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

!git clone https://github.com/spMohanty/PlantVillage-Dataset
!cd PlantVillage-Dataset

# a code to know the exact no. of images

import os
# Assuming the PlantVillage-Dataset directory is in the current working directory
!git clone https://github.com/spMohanty/PlantVillage-Dataset
dataset_path = 'PlantVillage-Dataset'  # Update with the correct path if different

if os.path.exists(dataset_path):
  image_count = 0
  for root, _, files in os.walk(dataset_path):
    for file in files:
      if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_count += 1
  print(f"Total number of images found in {dataset_path}: {image_count}")
else:
  print(f"Error: Directory '{dataset_path}' not found.")
