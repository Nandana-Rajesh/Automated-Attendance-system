import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import csv

# Load the trained CNN model
model_path = "trained_cnn_model.keras"  # Change this path if required
cnn_model = load_model(model_path)

# Path to the CSV file
csv_path = 'E:/Sem 3/Deep Learning/Project/Dataset 1/people.csv'
attendance_log_path = 'attendance_log.csv'

# Function to load class names from CSV
def load_class_names(csv_path):
    class_names = {}
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_names[int(row['images'])] = row['name']
    return class_names

# Load class names
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

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (128, 128))  # Resize face to match model input
            face_array = cv2.img_to_array(face_resized) / 255.0  # Normalize image
            face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

            # Predict the class label (i.e., identity)
            prediction = cnn_model.predict(face_array)
            predicted_label = np.argmax(prediction, axis=1)[0]

            # Generic message "Face Recognized"
            predicted_name = "Face Recognized"

            # Log attendance
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            attendance_log.append({'Name': predicted_name, 'Timestamp': timestamp, 'Status': 'Present'})

            # Display the "Face Recognized" message on the frame
            cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the frame with face detection and recognition
        cv2.imshow('Face Recognition Attendance System', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the attendance log to a CSV file
    attendance_df = pd.DataFrame(attendance_log)
    attendance_df.to_csv(attendance_log_path, index=False)
    print("Attendance logged to 'attendance_log.csv'.")

    # Release webcam and close OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Function to add new individuals
def add_new_person():
    name = entry_name.get()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return

    # Open the webcam to capture the person's image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        # Save the captured image
        image_path = f"E:/Sem 3/Deep Learning/Project/Dataset 1/preprocesses_image/{name}.jpg"
        cv2.imwrite(image_path, frame)

        # Update CSV with new name
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(class_names) + 1, name])  # Assuming image number is based on class count

        messagebox.showinfo("Success", f"Image for {name} saved successfully.")
    else:
        messagebox.showerror("Error", "Failed to capture image.")

    cap.release()

# Function to view attendance reports
def view_attendance():
    if os.path.exists(attendance_log_path):
        attendance_df = pd.read_csv(attendance_log_path)
        top = tk.Toplevel()
        top.title("Attendance Log")

        text_area = tk.Text(top)
        text_area.pack(expand=True, fill='both')
        text_area.insert(tk.END, attendance_df.to_string())
    else:
        messagebox.showerror("Error", "No attendance log found.")

# Tkinter GUI Setup
root = tk.Tk()
root.title("Face Recognition Attendance System")

# Add New Person Frame
frame_add_person = tk.Frame(root)
frame_add_person.pack(pady=10)

label_name = tk.Label(frame_add_person, text="Enter Name: ")
label_name.grid(row=0, column=0)

entry_name = tk.Entry(frame_add_person)
entry_name.grid(row=0, column=1)

button_add_person = tk.Button(frame_add_person, text="Add New Person", command=add_new_person)
button_add_person.grid(row=1, columnspan=2)

# View Attendance Button
button_view_attendance = tk.Button(root, text="View Attendance", command=view_attendance)
button_view_attendance.pack(pady=10)

# Monitor Attendance Button
button_monitor_attendance = tk.Button(root, text="Monitor Attendance", command=recognize_and_log_attendance)
button_monitor_attendance.pack(pady=10)

# Start the GUI loop
root.mainloop()
