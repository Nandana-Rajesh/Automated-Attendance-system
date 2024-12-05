from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

# Set the path to your model and test data
model_path = 'cnn_model_best.keras'  # Path to your trained model
test_dir = 'E:/Sem 3/Deep Learning/Project/Dataset 1/lfw-deepfunneled'  # Path to your test data

# Load the model
model = load_model(model_path)
print("Model loaded successfully.")

# Setup ImageDataGenerator for preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0, 1]

# Load the test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),  # Resize to match the input size of the model
    batch_size=32,
    class_mode='binary',  # For binary classification (two classes)
    shuffle=False  # We want to keep the order of the images for evaluation
)

# Get true labels (ground truth) and predicted labels
true_classes = test_generator.classes
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=1)

# Convert predictions to binary classes based on the threshold (0.5 for sigmoid output)
predicted_classes = (predictions >= 0.5).astype(int)

# Ensure the lengths match (in case there is a mismatch)
true_classes = true_classes[:len(predicted_classes)]  # Truncate true classes if there is any mismatch in length

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))
