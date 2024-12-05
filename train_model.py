from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model_path = "trained_cnn_model.keras"  # Change this to the correct model path
cnn_model = load_model(model_path)

print("Model loaded successfully.")

# Path to a test image (update the path to an actual test image)
test_image_path = "E:/Sem 3/Deep Learning/Project/Dataset 1/lfw-deepfunneled/lfw-deepfunneled/Alex_Sink/Alex_Sink_0001.jpg"  # Change this to the image path
print(f"Test image path: {test_image_path}")

# Load and preprocess the image
try:
    image = load_img(test_image_path, target_size=(128, 128))  # Resize to match model's input size
    image_array = img_to_array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the class of the image
    prediction = cnn_model.predict(image_array)
    print(f"Prediction: {prediction}")

    # Since we used sigmoid, the output is between 0 and 1, so apply a threshold to decide the class
    predicted_class = 1 if prediction >= 0.5 else 0
    print(f"Predicted class: {predicted_class}")

except FileNotFoundError:
    print(f"Test image not found: {test_image_path}")
