import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pandas as pd
import csv

# Load trained CNN model
model_path = "trained_cnn_model.keras"
cnn_model = load_model(model_path)

# Path to CSV
csv_path = 'E:/Sem 3/Deep Learning/Project/Dataset 1/people.csv'

def load_class_names(csv_path):
    class_names = {}
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                index = int(row['images'])
                class_names[index] = row['name']
            except ValueError:
                continue
    return class_names


class_names = load_class_names(csv_path)


st.title("Face Recognition Attendance")
image_input = st.camera_input("Open Camera to Monitor Attendance")

if image_input:
    # Convert Streamlit camera bytes to numpy array
    image_bytes = np.frombuffer(image_input, np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if frame is not None:
        # Resize the image and preprocess
        face_resized = cv2.resize(frame, (128, 128))
        face_array = face_resized / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Model Prediction
        prediction = cnn_model.predict(face_array)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_name = class_names.get(predicted_label, "Unknown")

        # Output results to the user
        st.write(f"Predicted: {predicted_name}")
    else:
        st.error("Error loading image data.")
else:
    st.warning("Waiting for camera input...")

