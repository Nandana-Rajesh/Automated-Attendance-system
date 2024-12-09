import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import gdown
import streamlit as st
from datetime import datetime
import csv

# Google Drive link for the model
gdrive_url = "https://drive.google.com/uc?id=165aJa711-e84lwd49c2GFJnO0zR4YWPe"

# Path to save the model locally
model_path = "trained_cnn_model.keras"
csv_path = 'people.csv'
attendance_log_path = 'attendance_log.csv'


# Function to download and load the model
def load_trained_model(model_path, gdrive_url):
    if not os.path.exists(model_path):
        try:
            st.info("Downloading model. Please wait...")
            gdown.download(gdrive_url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None

    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load class names from CSV
def load_class_names(csv_path):
    class_names = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    index = int(row['images'])
                    class_names[index] = row['name']
                except ValueError:
                    continue
    return class_names


# Recognize faces using the webcam
def recognize_face(cnn_model, class_names):
    cap = cv2.VideoCapture(0)
    detected_name = "Face not recognized"
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (128, 128))
            face_array = face_resized / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = cnn_model.predict(face_array)
            predicted_label = np.argmax(prediction, axis=1)[0]
            detected_name = class_names.get(predicted_label, "Face not recognized")
        else:
            st.error("No faces detected")

    cap.release()
    cv2.destroyAllWindows()
    return detected_name


# Streamlit Application
st.title("Face Recognition Attendance System")
st.write("### Mark Your Attendance")

# Load the model
cnn_model = load_trained_model(model_path, gdrive_url)
class_names = load_class_names(csv_path)

# Interface to mark attendance
name_to_mark = st.text_input("Enter Your Full Name")
if st.button("Mark Attendance"):
    if cnn_model:
        with st.spinner("Opening Camera to Detect Face..."):
            detected_name = recognize_face(cnn_model, class_names)
            if name_to_mark.strip() == detected_name:
                # Log attendance
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                attendance_entry = {'Name': detected_name, 'Timestamp': timestamp, 'Status': 'Present'}

                # Save to log file
                if os.path.exists(attendance_log_path):
                    attendance_df = pd.read_csv(attendance_log_path)
                else:
                    attendance_df = pd.DataFrame(columns=['Name', 'Timestamp', 'Status'])

                attendance_df = attendance_df.append(attendance_entry, ignore_index=True)
                attendance_df.to_csv(attendance_log_path, index=False)
                st.success(f"Attendance marked successfully for {detected_name}")
            else:
                st.error("Face does not match the entered name. Please try again.")

# View attendance log
if st.button("View Attendance Log"):
    if os.path.exists(attendance_log_path):
        st.write(pd.read_csv(attendance_log_path))
    else:
        st.error("No attendance log found.")

