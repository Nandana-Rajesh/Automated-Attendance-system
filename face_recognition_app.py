import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import csv

# Load the trained CNN model
model_path = "trained_cnn_model.keras"
cnn_model = load_model(model_path)

# Path to CSV and attendance log
csv_path = 'E:/Sem 3/Deep Learning/Project/Dataset 1/people.csv'
attendance_log_path = 'attendance_log.csv'


# Load class names safely
def load_class_names(csv_path):
    class_names = {}
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                index = int(row['images'])
                class_names[index] = row['name']
            except ValueError:
                continue  # Handle any invalid rows in CSV
    return class_names


# Function to recognize faces
def recognize_face():
    cap = cv2.VideoCapture(0)  # Open the webcam
    detected_name = "Face not recognized"
    ret, frame = cap.read()

    if ret:
        # Simulate face detection and recognition only on first frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Assuming only one face detected
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (128, 128))
            face_array = face_resized / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            # Prediction
            prediction = cnn_model.predict(face_array)
            predicted_label = np.argmax(prediction, axis=1)[0]
            detected_name = class_names.get(predicted_label, "Face not recognized")
        else:
            st.error("No faces detected")

    cap.release()
    cv2.destroyAllWindows()
    return detected_name


# Streamlit UI
st.title("Face Recognition Attendance System")
st.write("### Mark Your Attendance")

class_names = load_class_names(csv_path)

name_to_mark = st.text_input("Enter Your Full Name")
if st.button("Mark Attendance"):
    with st.spinner("Opening Camera to Detect Face..."):
        result = recognize_face()
        if name_to_mark.strip() in result:
            st.success(f"Attendance successfully marked for {name_to_mark}")
        else:
            st.error("Face not detected or unrecognized. Please ensure your name is trained in the model.")

if st.button("Monitor Attendance"):
    st.write("Starting live detection...")
    result = recognize_face()
    st.write(result)
 
