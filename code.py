The complete code that used in the project is provided below:-
# Install Kaggle and Importing necessary libraries
!pip install kaggle
import json
import zipfile
import shutil
import random  # For random sampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # For loading and processing images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import itertools




# Load Kaggle credentials from the saved JSON file
with open('kaggle.json') as file:
    kaggle_credentials = json.load(file)

# Set Kaggle username and key as environment variables
os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
os.environ['KAGGLE_KEY'] = kaggle_credentials['key']
     

# Download the dataset
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset
# Dataset source:https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data


# Creating a directory where we extract the dataset
dataset_folder = '/content/plant_disease_dataset'
os.makedirs(dataset_folder, exist_ok=True)

# Unzip the downloaded file into our target directory
with zipfile.ZipFile('new-plant-diseases-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall(dataset_folder)

# Path to the wrong duplicate folder
wrong_folder = '/content/plant_disease_dataset/new plant diseases dataset(augmented)'

# Delete it
shutil.rmtree(wrong_folder)

print("Duplicate folder deleted successfully.")
     

!ls


# Let's quickly check what files and folders are inside the extracted dataset
print(os.listdir('/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)'))


# lets Setting up the base paths for training and validation datasets
train_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

# Let's list out all the classes (folders) available in the training set
train_classes = os.listdir(train_dir)

# Printing how many different classes we have
print(f"Total number of classes in the training set: {len(train_classes)}")

# Just checking a few sample class names to get an idea
print("Sample class names:", train_classes[:5])


EDA
# Now, let's see how many images we have for each class in the training set
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in train_classes}

print("\nNumber of images per class:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")

# Let's plot the class distribution to get a visual idea
plt.figure(figsize=(18, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))

# Rotating x-axis labels for better readability
plt.xticks(rotation=90)

# Adding a title and labels to make the plot clear
plt.title('Training Set - Class Distribution')
plt.xlabel('Class Names')
plt.ylabel('Number of Images')

# Finally, displaying the plot
plt.show()
# Let's display some random sample images from the training set to get a feel for the dataset


# try to gather all the image paths along with their class names
train_classes = os.listdir(train_dir)
all_image_paths = []

for cls in train_classes:
    cls_folder = os.path.join(train_dir, cls)
    for img_file in os.listdir(cls_folder):
        all_image_paths.append((os.path.join(cls_folder, img_file), cls))

# Randomly pick 9 images to visualize
random_samples = random.sample(all_image_paths, 9)

# Set up a 3x3 grid for displaying the images
plt.figure(figsize=(15, 15))

for idx, (img_path, cls_name) in enumerate(random_samples):
    img = image.load_img(img_path, target_size=(224, 224))
    plt.subplot(3, 3, idx + 1)
    plt.imshow(img)
    plt.title(cls_name, fontsize=10)
    plt.axis('off')  # Hide axes for cleaner look

# Adjust layout so titles don't overlap
plt.tight_layout(pad=3.0)

# Add a main title above all images
plt.subplots_adjust(top=0.88)
plt.suptitle('Random 9 Sample Images - Plant Disease Dataset', fontsize=20)
plt.show()

# Setting up the paths for training and validation datasets
train_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

# Let's count the number of images per class in both training and validation sets
train_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)}
valid_counts = {cls: len(os.listdir(os.path.join(valid_dir, cls))) for cls in os.listdir(valid_dir)}

# Now, converting the counts into a DataFrame for easier visualization
df_counts = pd.DataFrame({
    'Class': list(train_counts.keys()),
    'Train Images': list(train_counts.values()),
    'Validation Images': [valid_counts.get(cls, 0) for cls in train_counts.keys()]  # Just in case a class is missing
})

# Let's sort the DataFrame based on the number of training images (makes the plot cleaner)
df_counts_sorted = df_counts.sort_values('Train Images', ascending=False)

# Plotting the class distribution for both train and validation sets
plt.figure(figsize=(20, 12))
bar_width = 0.4

r1 = range(len(df_counts_sorted))
r2 = [x + bar_width for x in r1]  # Offset for validation bars

# Create side-by-side bars
plt.bar(r1, df_counts_sorted['Train Images'], width=bar_width, label='Train Set', col-or='skyblue', edgecolor='black')
plt.bar(r2, df_counts_sorted['Validation Images'], width=bar_width, label='Validation Set', color='lightgreen', edgecolor='black')

# Adding labels and title
plt.xlabel('Class Names', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Train vs Validation Set Class Distribution', fontsize=16)

# Setting x-ticks to be between the two bars
plt.xticks([r + bar_width/2 for r in range(len(df_counts_sorted))], df_counts_sorted['Class'], rotation=90)

# Adding a legend to differentiate
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Finally, show the plot
plt.show()

 
Figure A: Class Distribution in Training and Validation Sets
This bar chart compares the number of images per class in the training and validation sets. Each class (representing a specific plant disease or healthy condition) has a balanced number of images, ensuring fair representation during model training and evaluation. The dataset fol-lows an approximate 80/20 split, which is ideal for deep learning tasks.


#
# Setting the training directory path
train_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'

# Counting the number of images for each class
train_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)}

# Converting the counts dictionary into a DataFrame for easier plotting
train_counts_df = pd.DataFrame(list(train_counts.items()), columns=['Class', 'Number of Im-ages'])

# Sorting classes by the number of images (ascending) to spot imbalances easily
train_counts_df_sorted = train_counts_df.sort_values('Number of Images', ascending=True)

# Plotting the number of images per class
plt.figure(figsize=(14, 10))
sns.barplot(data=train_counts_df_sorted, x='Number of Images', y='Class', palette='viridis')

# Adding title and axis labels
plt.title('Training Set - Class Imbalance Check', fontsize=16)
plt.xlabel('Number of Images', fontsize=14)
plt.ylabel('Class Names', fontsize=14)

# Adding a grid along x-axis for better visual alignment
plt.grid(axis='x')

# Adjust layout to make sure everything fits nicely
plt.tight_layout()

# Show the plot
plt.show()

 

Figure B: Training Set – Class Imbalance Check:- This horizontal bar chart displays the number of images per class in the training dataset, helping visualize any potential class imbalance. While most classes are relatively well-balanced, a few have slightly fewer samples. Identifying such variations is important to ensure the model does not favour overrepresented classes dur-ing training.



Data Agmentation:-
# Let's define a simple ImageDataGenerator with some basic augmentations
augmenter = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Randomly pick an image from the training set to visualize augmentations
train_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
random_class = random.choice(os.listdir(train_dir))
random_img_path = os.path.join(train_dir, random_class, ran-dom.choice(os.listdir(os.path.join(train_dir, random_class))))

# Load and preprocess the selected image
img = load_img(random_img_path, target_size=(224, 224))  # Resizing to match model input size
img_array = img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to feed into the augmenter

# Generate augmented images
aug_iter = augmenter.flow(img_array, batch_size=1)

# Let's plot the original image along with 8 augmented versions
plt.figure(figsize=(15, 15))

# Display the original image
plt.subplot(3, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Display 8 augmented images
for i in range(2, 10):
    aug_img = next(aug_iter)[0]
    plt.subplot(3, 3, i)
    plt.imshow(aug_img)
    plt.title(f'Augmented {i-1}')
    plt.axis('off')

# Adjust layout and add a main title
plt.tight_layout()
plt.suptitle('Data Augmentation Examples', fontsize=20, y=1.02)
plt.show()

 

Figure C: Data Augmentation Examples:- This grid shows an original leaf image (top-left) fol-lowed by eight augmented versions generated through transformations such as rotation, shift-ing, and zooming. Data augmentation increases the diversity of the training dataset, helping the model generalize better and reduce overfitting.





RGB Color Channel Analysis of Sample Images:-
# Let's pick a random image from the training set for RGB channel analysis
train_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
random_class = random.choice(os.listdir(train_dir))
random_img_path = os.path.join(train_dir, random_class, ran-dom.choice(os.listdir(os.path.join(train_dir, random_class))))

# Load the selected image and resize it to (224, 224)
img = load_img(random_img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0,1] range

# Split the image into Red, Green, and Blue channels
R = img_array[:, :, 0]
G = img_array[:, :, 1]
B = img_array[:, :, 2]

# Plotting the original image along with its R, G, B channels
plt.figure(figsize=(12, 10))

# Display original image
plt.subplot(2, 2, 1)
plt.imshow(img_array)
plt.title('Original Image')
plt.axis('off')

# Display Red channel
plt.subplot(2, 2, 2)
plt.imshow(R, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

# Display Green channel
plt.subplot(2, 2, 3)
plt.imshow(G, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

# Display Blue channel
plt.subplot(2, 2, 4)
plt.imshow(B, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

# Add a main title and adjust layout
plt.suptitle('RGB Color Channel Analysis', fontsize=18)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

      

Figure D: RGB Color Channel Analysis:- This visualization breaks down a leaf image into its three RGB color channels. The original image is shown at the top left, followed by the red, green, and blue components. Analyzing these channels can help reveal subtle visual patterns, such as disease spots or texture differences, which might be more prominent in certain channels and assist in better feature extraction for model training.


Loading and Preparing Training, Validation, and Test Datasets:-
# Define the paths for training, validation, and test datasets
train_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'
test_dir = '/content/plant_disease_dataset/test'  # Assuming the test set is organized correctly

# Load the training dataset
training_set = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',             # Automatically use folder names as labels
    label_mode='categorical',       # One-hot encode the labels
    batch_size=32,                  # Number of images per batch
    image_size=(128, 128),           # Resize images to 128x128
    shuffle=True                    # Shuffle the dataset for better training
)

# Load the validation dataset
validation_set = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

# Load the test dataset
# Note: Shuffling is set to False to maintain the original order during evaluation
test_set = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(128, 128),
    shuffle=False
)
     

# Let's quickly check how many classes we have and what they are
print(f"Total number of classes: {len(training_set.class_names)}")
print(f"Class labels: {training_set.class_names}")


# Let's take a quick look at one batch from the training set
for x_batch, y_batch in training_set.take(1):
    print("Batch image shape:", x_batch.shape)
    print("Batch label shape:", y_batch.shape)


# Just checking one full batch to understand the structure of images and labels
for x, y in training_set:
    print(x)
    print("Image batch shape:", x.shape)
    print(y)
    print("Label batch shape:", y.shape)
    break  # Only checking the first batch

#Lets Build a model:-

# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Initialize a sequential model
model = Sequential()

# First convolutional block
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(128,128,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Second convolutional block
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Third convolutional block
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Fourth convolutional block
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Fifth convolutional block
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Dropout layer to reduce overfitting
model.add(Dropout(0.25))

# Flatten the output from convolutional blocks
model.add(Flatten())

# Fully connected dense layer
model.add(Dense(units=1500, activation='relu'))

# Additional dropout layer
model.add(Dropout(0.4))

# Output layer with softmax activation (38 classes)
model.add(Dense(units=38, activation='softmax'))

# Compile model with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture summary
model.summary()
# Import necessary callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define EarlyStopping callback to stop training if validation loss doesn't improve
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Define ModelCheckpoint callback to save the best model during training
checkpoint = ModelCheckpoint(
    'best_plant_disease_model.h5',  # file path to save the model
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model with defined callbacks
history = model.fit(
    training_set,                  # previously prepared training set
    epochs=10,                     # number of epochs (adjustable)
    validation_data=validation_set,  # previously prepared validation set
    callbacks=[early_stop, checkpoint]
)
#Model Evaluation on Training set
train_loss,train_acc = model.evaluate(training_set)

# Print the training loss and accuracy
print(train_loss, train_acc)
     
# Evaluate model on validation set
val_loss, val_acc = model.evaluate(validation_set)

# Print results
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

Plotting the curves:-
# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Prepare test dataset
test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False
)

y_pred = model.predict(test_set)
y_pred,y_pred.shape
     

predicted_categories = tf.argmax(y_pred,axis=1)
predicted_categories
     

true_categories = tf.concat([y for x,y in test_set],axis=0)
true_categories
     

Y_true = tf.argmax(true_categories,axis=1)
Y_true
from sklearn.metrics import classification_report,confusion_matrix
# Generate and print classification report
print(classification_report(Y_true, predicted_categories, target_names=list(training_set.class_names)))

# Predict classes on the test set
y_pred = model.predict(test_set)
y_pred_classes = np.argmax(y_pred, axis=1)

# Extract true labels
y_true = np.concatenate([y for x, y in test_set], axis=0)
y_true_classes = np.argmax(y_true, axis=1)

# Class names
class_names = list(test_set.class_names)

# Generate and print classification report
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Compute and plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(20,18))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=22)
plt.xlabel('Predicted Class', fontsize=18)
plt.ylabel('True Class', fontsize=18)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
2.	Model 2: Transfer Learning — MobileNetV2

#Again Load Training and Validation Sets
train_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_dir = '/content/plant_disease_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

# Load the training dataset
training_set = tf.keras.utils.image_dataset_from_directory(
    train_dir,  # Path to the training data directory
    labels='inferred',  # Labels are inferred from the directory structure
    label_mode='categorical',  # One-hot encode the labels
    batch_size=32,  # Number of images in each batch
    image_size=(128, 128),  # Resize all images to 128x128 pixels
    shuffle=True  # Shuffle the data to prevent bias
)

# Load the validation dataset
validation_set = tf.keras.utils.image_dataset_from_directory(
    valid_dir,  # Path to the validation data directory
    labels='inferred',  # Labels are inferred from the directory structure
    label_mode='categorical',  # One-hot encode the labels
    batch_size=32,  # Number of images in each batch
    image_size=(128, 128),  # Resize all images to 128x128 pixels
    shuffle=True  # Shuffle the data to prevent bias
)
# Enable prefetching to optimize the input pipeline
AUTOTUNE = tf.data.AUTOTUNE  # Automatically tune the buffer size for optimal performance

# Apply prefetching to the training dataset
training_set = training_set.prefetch(buffer_size=AUTOTUNE)

# Apply prefetching to the validation dataset
validation_set = validation_set.prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

#2nd Model:- Build MobileNetV2 Model

# Load the MobileNetV2 model with pre-trained ImageNet weights, excluding the top layer
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # Input size of the images (128x128, RGB)
    include_top=False,  # Exclude the final fully connected layers
    weights='imagenet'  # Use pre-trained weights from ImageNet
)

# Freeze the base model (we don't want to train these layers initially)
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output  # Get the output of the base model
x = GlobalAveragePooling2D()(x)  # Apply global average pooling
x = Dense(512, activation='relu')(x)  # Fully connected layer with 512 units and ReLU activation
x = Dropout(0.5)(x)  # Dropout layer with a 50% chance of dropout
predictions = Dense(38, activation='softmax')(x)  # Output layer with 38 classes (softmax for multi-class classification)

# Create the final model
mobilenet_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with Adam optimizer and categorical crossentropy loss
mobilenet_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model summary to see the architecture
mobilenet_model.summary()
     from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stopping: stops training if the validation loss doesn't improve for 5 consecutive epochs
early_stop_mobilenet = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop after 5 epochs with no improvement
    restore_best_weights=True,  # Restore the best weights when training stops
    verbose=1  # Print messages when training stops early
)

# Model checkpoint: save the best model based on validation loss during training
checkpoint_mobilenet = ModelCheckpoint(
    'best_mobilenet_model.h5',  # File name to save the model
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model
    verbose=1  # Print messages when saving the best model
)

# Train the model with only the top layers
history_initial = mobilenet_model.fit(
    training_set,  # Training dataset
    validation_data=validation_set,  # Validation dataset
    epochs=10,  # Number of epochs for training
    callbacks=[early_stop_mobilenet, checkpoint_mobilenet]  # Callbacks to monitor performance
)

# Fine-tune the MobileNetV2 model by unfreezing some layers
base_model.trainable = True  # Allow training of the base model

# Freeze the first 100 layers of the base model to keep their weights fixed
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile the model with a smaller learning rate for fine-tuning
mobilenet_model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)
# Fine-tune the full MobileNetV2 model
history_finetune = mobilenet_model.fit(
    training_set,  # Training dataset
    validation_data=validation_set,  # Validation dataset
    epochs=20,  # Number of epochs for fine-tuning
    callbacks=[early_stop_mobilenet, checkpoint_mobilenet]  # Callbacks for early stopping and saving the best model
)
 # evaluate the model on the training and validation datasets
train_loss, train_acc = mobilenet_model.evaluate(training_set)  # Evaluate on training set
val_loss, val_acc = mobilenet_model.evaluate(validation_set)  # Evaluate on validation set

# printing evaluation results
print(f"\nTrain Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
# Plot training and validation accuracy
plt.figure(figsize=(10,6))
plt.plot(history_initial.history['accuracy'] + history_finetune.history['accuracy'], label='Training Accuracy')
plt.plot(history_initial.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10,6))
plt.plot(history_initial.history['loss'] + history_finetune.history['loss'], label='Training Loss')
plt.plot(history_initial.history['val_loss'] + history_finetune.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
     # Confusion Matrix for mobilenet model

# Reload validation set (shuffling is disabled to keep the true order of the data)
test_set = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(128, 128),
    shuffle=False
)

# Get the class names
class_names = test_set.class_names

# Get model predictions
y_pred = mobilenet_model.predict(test_set)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true labels
y_true = np.concatenate([y for x, y in test_set], axis=0)
y_true_classes = np.argmax(y_true, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(22, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - MobileNetV2 Fine-tuned', fontsize=24)
plt.xlabel('Predicted Class', fontsize=18)
plt.ylabel('True Class', fontsize=18)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# Print the classification report for the model
print("\nClassification Report:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Get one batch of images and labels from the validation/test set
images, labels = next(iter(test_set))

# Make predictions on the batch
predictions = mobilenet_model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(labels.numpy(), axis=1)

# Get class names
class_names = list(test_set.class_names)

# Plot random 9 images and their predicted vs true labels
plt.figure(figsize=(15, 15))
for i in range(9):
    index = np.random.randint(0, len(images))  # Pick a random image index
    plt.subplot(3, 3, i+1)
    plt.imshow(images[index].numpy().astype("uint8"))

    # Get predicted and true labels
    pred_label = class_names[predicted_classes[index]]
    true_label = class_names[true_classes[index]]

    # Color the title green if prediction is correct, red if incorrect
    color = "green" if pred_label == true_label else "red"

    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
    plt.axis('off')  # Hide axis for better visual

plt.suptitle('Random Sample Predictions - MobileNetV2 Fine-tuned', fontsize=20)
plt.show()

3rd Model VGG16
### again importing libraries

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
# Load the VGG16 base model without the top layer
base_vgg16 = VGG16(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
base_vgg16.trainable = False

# Add custom layers on top of VGG16
x = base_vgg16.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(38, activation='softmax')(x)

# Create the final model
vgg16_model = Model(inputs=base_vgg16.input, outputs=outputs)

# Compile the model
vgg16_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show the model summary
vgg16_model.summary()

# Early stopping and checkpoint callbacks
early_stop_vgg16 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint_vgg16 = tf.keras.callbacks.ModelCheckpoint('best_vgg16_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
history_vgg16 = vgg16_model.fit(
    training_set,  # Training dataset
    epochs=20,  # Number of epochs
    validation_data=validation_set,  # Validation dataset
    callbacks=[early_stop_vgg16, checkpoint_vgg16]  # Callbacks for early stopping and saving the best model
)

# Accuracy plot
plt.figure(figsize=(12,6))
plt.plot(history_vgg16.history['accuracy'], label='Training Accuracy')
plt.plot(history_vgg16.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy - VGG16')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss plot
plt.figure(figsize=(12,6))
plt.plot(history_vgg16.history['loss'], label='Training Loss')
plt.plot(history_vgg16.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss - VGG16')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
# Evaluate the model on the training and validation sets
train_loss_vgg16, train_acc_vgg16 = vgg16_model.evaluate(training_set)
val_loss_vgg16, val_acc_vgg16 = vgg16_model.evaluate(validation_set)

# Print the evaluation results
print(f"Training Accuracy: {train_acc_vgg16:.4f}")
print(f"Validation Accuracy: {val_acc_vgg16:.4f}")
# Make predictions on the test set using the trained model
y_pred = vgg16_model.predict(test_set)  # Using VGG16 model for predictions
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class labels

# get the true labels from the test set
y_true = np.concatenate([y for x, y in test_set], axis=0)
y_true_classes = np.argmax(y_true, axis=1)  # Get true class labels

# get the class names (labels) from the test set
class_names = list(test_set.class_names)

#calculate the confusion matrix
cm_vgg16 = confusion_matrix(y_true_classes, y_pred_classes)

#plot the confusion matrix
plt.figure(figsize=(20, 18))
sns.heatmap(cm_vgg16, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - VGG16', fontsize=22)
plt.xlabel('Predicted Class', fontsize=18)
plt.ylabel('True Class', fontsize=18)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

#print the classification report
print("\nClassification Report:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

#get one batch of images and labels from the test/validation set
images, labels = next(iter(test_set))

#make predictions on the batch using the VGG16 model
predictions = vgg16_model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(labels.numpy(), axis=1)

#get class names from the test set
class_names = list(test_set.class_names)

#plot random 9 images with their true and predicted labels
plt.figure(figsize=(15, 15))
for i in range(9):
    index = np.random.randint(0, len(images))  # Pick a random image index
    plt.subplot(3, 3, i+1)
    plt.imshow(images[index].numpy().astype("uint8"))  # Display the image
    pred_label = class_names[predicted_classes[index]]  # Predicted label
    true_label = class_names[true_classes[index]]  # True label


    color = "green" if pred_label == true_label else "red"  # Color the title green if prediction is correct, red if incorrect

    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
    plt.axis('off')  # Turn off axis for cleaner display

# Add a title to the plot
plt.suptitle('Random Sample Predictions - VGG16', fontsize=20)
plt.show()
#evaluate Base CNN --1st model
base_train_loss, base_train_acc = model.evaluate(training_set)
base_val_loss, base_val_acc = model.evaluate(validation_set)

print("\nBase CNN Results:")
print(f"Training Accuracy: {base_train_acc:.4f}, Training Loss: {base_train_loss:.4f}")
print(f"Validation Accuracy: {base_val_acc:.4f}, Validation Loss: {base_val_loss:.4f}")

#evaluate MobileNetV2 Fine-tuned Model --- 2nd model
mobilenet_train_loss, mobilenet_train_acc = mobilenet_model.evaluate(training_set)
mobilenet_val_loss, mobilenet_val_acc = mobilenet_model.evaluate(validation_set)

print("\nMobileNetV2 Fine-tuned Results:")
print(f"Training Accuracy: {mobilenet_train_acc:.4f}, Training Loss: {mobilenet_train_loss:.4f}")
print(f"Validation Accuracy: {mobilenet_val_acc:.4f}, Validation Loss: {mobilenet_val_loss:.4f}")

# evaluate VGG16 Model --- 3rd model
vgg16_train_loss, vgg16_train_acc = vgg16_model.evaluate(training_set)
vgg16_val_loss, vgg16_val_acc = vgg16_model.evaluate(validation_set)

print("\nVGG16 Results:")
print(f"Training Accuracy: {vgg16_train_acc:.4f}, Training Loss: {vgg16_train_loss:.4f}")
print(f"Validation Accuracy: {vgg16_val_acc:.4f}, Validation Loss: {vgg16_val_loss:.4f}")

#Below are the models prepared
models = ['Base CNN', 'MobileNetV2', 'VGG16']

# Training and validation accuracies for each model
train_accuracy = [0.9915, 0.9975, 0.9929]
val_accuracy = [0.9674, 0.9028, 0.9577]

# Training and validation losses for each model
train_loss = [0.0265, 0.0155, 0.0315]
val_loss = [0.1144, 0.3319, 0.1283]

# Create a figure for the accuracy comparison
plt.figure(figsize=(10, 6))

# Set the positions for the bars
x = np.arange(len(models))
width = 0.3  # Width of the bars

# Plot the bars for training and validation accuracy
plt.bar(x - width/2, train_accuracy, width, label='Training Accuracy', color='blue')
plt.bar(x + width/2, val_accuracy, width, label='Validation Accuracy', color='orange')

# Add the accuracy values on top of the bars for clarity
for bars in [plt.bar(x - width/2, train_accuracy, width), plt.bar(x + width/2, val_accuracy, width)]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)

# Set the labels, title, and formatting for the accuracy plot
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Model Comparison: Training vs Validation Accuracy', fontsize=16, pad=20)
plt.xticks(x, models, fontsize=12)
plt.yticks(np.arange(0.85, 1.01, 0.02))
plt.ylim(0.85, 1.00)
plt.legend(fontsize=12)
plt.grid(axis='y')

# Display the accuracy plot
plt.tight_layout()
plt.show()

# Create a figure for the loss comparison
plt.figure(figsize=(10, 6))

# Plot the bars for training and validation loss
plt.bar(x - width/2, train_loss, width, label='Training Loss', color='blue')
plt.bar(x + width/2, val_loss, width, label='Validation Loss', color='orange')

# Add the loss values on top of the bars for clarity
for bars in [plt.bar(x - width/2, train_loss, width), plt.bar(x + width/2, val_loss, width)]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)

# Set the labels, title, and formatting for the loss plot
plt.xlabel('Models', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Model Comparison: Training vs Validation Loss', fontsize=16, pad=20)
plt.xticks(x, models, fontsize=12)
plt.yticks(np.arange(0, 0.35, 0.02))
plt.ylim(0, 0.35)
plt.legend(fontsize=12)
plt.grid(axis='y')

# Display the loss plot
plt.tight_layout()
plt.show()
