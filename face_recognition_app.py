import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import csv


# Path to trained model
model_path = "E:/Sem 3/Deep Learning/Project/Dataset 1/trained_cnn_model.keras"

# Ensure the model exists
try:
    cnn_model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model. Ensure path is correct. Details: {e}")


# Path to CSV and attendance log
csv_path = 'E:/Sem 3/Deep Learning/Project/Dataset 1/people.csv'
attendance_log_path = 'attendance_log.csv'


# Load class names safely
def load_class_names(csv_path):
    class_names = {}
    try:
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    index = int(row['images'])
                    class_names[index] = row['name']
                except ValueError:
                    continue  # Handle invalid rows
    except Exception as e:
        st.error(f"Error loading CSV. Details: {e}")
    return class_names


# Function to recognize faces via webcam and predict names
def recognize_face():
    cap = cv2.VideoCapture(0)  # Open the webcam
    detected_name = "Face not recognized"
    ret, frame = cap.read()

    if ret:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Focus on the first detected face
            (x, y, w, h) = faces[0]
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (128, 128))  # Resize the face image
            face_array = face_resized / 255.0
            face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

            # Predict using the model
            try:
                prediction = cnn_model.predict(face_array)
                predicted_label = np.argmax(prediction, axis=1)[0]
                class_names = load_class_names(csv_path)
                detected_name = class_names.get(predicted_label, "Face not recognized")
            except Exception as e:
                st.error(f"Error predicting face. Details: {e}")

    cap.release()
    cv2.destroyAllWindows()
    return detected_name


# Streamlit UI
st.title("Face Recognition Attendance System")
st.write("### Mark Your Attendance")

# Input for entering name
name_to_mark = st.text_input("Enter Your Full Name:")
if st.button("Mark Attendance"):
    with st.spinner("Attempting to detect face..."):
        result = recognize_face()
        if name_to_mark.strip() in result:
            st.success(f"Attendance successfully marked for {name_to_mark}")
        else:
            st.error("Face not detected or unrecognized. Ensure your name is trained in the model.")

if st.button("Monitor Attendance"):
    st.write("Starting live detection...")
    with st.spinner("Monitoring webcam..."):
        result = recognize_face()
        st.write(f"Detected result: {result}")
