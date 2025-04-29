# Pest Detection Using Deep Learning (Crop Images)

## 📌 Overview
This project aims to classify crop leaf images to detect pest infections and plant diseases using deep learning. It uses the **New Plant Diseases Dataset**, which contains around **87,000 images** of healthy and diseased crop leaves across **38 classes**. The goal is to support early disease detection in agriculture, helping reduce crop loss and improve productivity.

## 🎓 About This Project
This project was completed as part of my MSc dissertation at the **University of Hertfordshire**. The aim was to explore the use of deep learning for identifying plant diseases and pest symptoms using image classification models.

## 📁 Dataset
- **Source**: [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Total Images**: ~87,000 RGB images
- **Classes**: 38 (including both healthy and diseased)
- **Split**: 80% training, 20% validation, 33 test images

## 🤖 Models Used

| Model        | Description                                  |
|--------------|----------------------------------------------|
| **Base CNN** | A simple CNN built from scratch              |
| **MobileNetV2** | A fast and lightweight pre-trained model    |
| **VGG16**       | A deep and powerful pre-trained model       |

## ✅ Key Results

| Model         | Validation Accuracy | Notes                                 |
|---------------|---------------------|----------------------------------------|
| Base CNN      | 96.74%              | Strong performance, basic model        |
| VGG16         | 95.77%              | High accuracy, slower training         |
| MobileNetV2   | 90.28%              | Lightweight and fast, less accurate    |

## ⚙️ Setup & Installation

To run this project locally:

```bash
git clone https://github.com/Deepakvishwakarma1/Pest-Detection-Using-Deep-Learning_Crop-Images_Major_project.git
cd Pest-Detection-Using-Deep-Learning_Crop-Images_Major_project
pip install -r requirements.txt
