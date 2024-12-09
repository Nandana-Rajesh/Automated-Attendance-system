import os
import requests
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from datetime import datetime


# Path for the model
model_path = "trained_cnn_model.keras"
google_drive_url = "https://drive.google.com/uc?id=165aJa711-e84lwd49c2GFJnO0zR4YWPe"  # Replace with your model's direct URL


# Function to download the model file if it's not already present
def download_model(url, output_path):
    if not os.path.exists(output_path):
        st.info("Model file not found. Downloading...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                st.success("Model downloaded successfully.")
            else:
                st.error(f"Failed to download the model. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error while downloading the model: {e}")
    else:
        st.success("Model file found locally.")


# Load the trained CNN model
def load_trained_model():
    # Download the model if not present
    download_model(google_drive_url, model_path)

    # Load model
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# Initialize Streamlit App
st.title("Face Recognition Attendance System")
st.write("### Mark Your Attendance with Face Recognition")

# Load the trained model
cnn_model = load_trained_model()


# Function to recognize a face from webcam
def recognize_face(cnn_model):
    cap = cv2.VideoCapture(0)  # Start webcam
    detected_name = "Face not recognized"
    try:
        st.info("Camera is accessing...")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            # Convert the frame to grayscale for detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (128, 128))
                face_array = face_resized / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                # Predict using the trained CNN model
                prediction = cnn_model.predict(face_array)
                predicted_label = np.argmax(prediction, axis=1)[0]

                # Map the label to the name (ensure class_names are loaded beforehand)
                predicted_name = class_names.get(predicted_label, "Unknown")
                st.write(f"Face Detected: {predicted_name}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                st.info("No faces detected.")

            # Display the image
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        st.error(f"Error while accessing the webcam: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Allow user to mark attendance by entering name
user_name = st.text_input("Enter Your Full Name:")

if st.button("Mark Attendance"):
    if cnn_model:
        with st.spinner("Starting camera for face recognition..."):
            recognize_face(cnn_model)
    else:
        st.error("Model is not loaded, cannot recognize face.")


