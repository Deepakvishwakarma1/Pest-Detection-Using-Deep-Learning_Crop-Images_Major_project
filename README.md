
---
# Pest Detection Using Deep Learning (Crop Images)

## Overview
This project aims to classify crop leaf images to detect pest infections and plant diseases using deep learning techniques. It uses the **New Plant Diseases Dataset**, which contains around **87,000 images** of healthy and diseased leaves across **38 different classes**.

The goal is to support early disease detection in agriculture, which can help reduce crop loss and improve productivity.

---

## About This Project
This work is part of my MSc dissertation at the **University of Hertfordshire**. The main objective was to evaluate the performance of different deep learning models in identifying crop diseases and pest symptoms using image classification.

---

## Dataset
- **Source**: [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: 38 (including both healthy and diseased)
- **Images**: ~87,000
- **Split**: 80% training / 20% validation / separate test set

---

## Models Used
| Model        | Description                                |
|--------------|--------------------------------------------|
| **Base CNN** | A custom Convolutional Neural Network built from scratch. |
| **MobileNetV2** | A fast and lightweight pre-trained model. |
| **VGG16**       | A deeper and more accurate pre-trained model. |

---

## Key Results

| Model        | Validation Accuracy | Notes                                 |
|--------------|---------------------|----------------------------------------|
| Base CNN     | 96.74%              | Best performance overall               |
| VGG16        | 95.77%              | Very accurate but slower to train      |
| MobileNetV2  | 90.28%              | Fast, but didn’t perform as well here  |

## Setup & Installation

To run the project locally:

```bash
git clone https://github.com/Deepakvishwakarma1/Pest-Detection-Using-Deep-Learning_Crop-Images_Major_project.git
cd Pest-Detection-Using-Deep-Learning_Crop-Images_Major_project
pip install -r requirements.txt



