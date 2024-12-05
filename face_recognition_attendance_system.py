import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
import csv

# Load the trained CNN model
model_path = "trained_cnn_model.keras"  # Change this path if required
cnn_model = load_model(model_path)

# Load class names (mapping image numbers to names)
def load_class_names(csv_path):
    class_names = {}
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_names[int(row['images'])] = row['name']  # Adjust CSV field names as necessary
    return class_names

# Path to the CSV that links images to names
csv_path = 'E:/Sem 3/Deep Learning/Project/Dataset 1/people.csv'  # Adjust path if needed
class_names = load_class_names(csv_path)

# Initialize OpenCV for face detection (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize face and log attendance
def recognize_and_log_attendance():
    cap = cv2.VideoCapture(0)  # Start webcam
    attendance_log = []  # List to log attendance

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

        # For each detected face, perform recognition
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (128, 128))  # Resize face to match model input
            face_array = img_to_array(face_resized) / 255.0  # Normalize image
            face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

            # Predict the class label (i.e., identity)
            prediction = cnn_model.predict(face_array)
            predicted_label = np.argmax(prediction, axis=1)[0]

            # Get the person's name from the class names
            predicted_name = class_names.get(predicted_label, "Face Recognised")

            # Log attendance
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            attendance_log.append({'Name': predicted_name, 'Timestamp': timestamp, 'Status': 'Present'})

            # Display the name and face on the frame
            cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the frame with face detection and recognition
        cv2.imshow('Face Recognition Attendance System', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the attendance log to a CSV file
    attendance_df = pd.DataFrame(attendance_log)
    attendance_df.to_csv('attendance_log.csv', index=False)
    print("Attendance logged to 'attendance_log.csv'.")

    # Release webcam and close OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Run face recognition and logging
recognize_and_log_attendance()

