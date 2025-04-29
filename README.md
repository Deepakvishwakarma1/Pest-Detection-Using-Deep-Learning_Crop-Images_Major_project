# Plantvillage_dataset_Major_project changed to New Plant Diseases Dataset
Pest Detection Using Deep Learning(Crop Images)
#  Plant Disease and Pest Detection Using Deep Learning

## About This Project

This project is my MSc dissertation work for the University of Hertfordshire.  
The main idea is to use deep learning to identify **plant diseases and pest infections** by analyzing images of crop leaves. Early detection is really important in agriculture, and I wanted to explore how AI can help with that.

I used a dataset from Kaggle called the **New Plant Diseases Dataset**, which includes around **87,000 images** of healthy and unhealthy leaves. The images cover **38 different classes**, and I trained several deep learning models to see which one could make the most accurate predictions.

---

##  Models I Used

- **Base CNN** – A simple model that I built myself from scratch
- **MobileNetV2** – A fast and lightweight pre-trained model
- **VGG16** – A more complex and deeper pre-trained model

---

##  What I Did

- Resized and normalized the images
- Applied **data augmentation** (like flipping, zooming, rotating) to help the model learn better
- Trained the models using an **80/20 train-validation split**
- Used **early stopping** to avoid overfitting
- Compared model results using accuracy, loss, and confusion matrices

---

## Key Results

| Model        | Validation Accuracy | Notes                                 |
|--------------|---------------------|----------------------------------------|
| Base CNN     | 96.74%              | Best performance overall               |
| VGG16        | 95.77%              | Very accurate but slower to train      |
| MobileNetV2  | 90.28%              | Fast, but didn’t perform as well here  |

---

## What's in This Repository

