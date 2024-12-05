from PIL import Image
import os

# Paths
dataset_path = r"E:\Sem 3\Deep Learning\Project\Dataset 1\lfw-deepfunneled\lfw-deepfunneled"
output_path = r"E:\Sem 3\Deep Learning\Project\Dataset 1\preprocesses_image"

# Parameters
target_size = (128, 128)  # Desired size for resizing

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Processing images
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_filename)

        try:
            # Open image
            img = Image.open(image_path).convert("RGB")

            # Resize with LANCZOS resampling
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Save preprocessed image
            save_folder = os.path.join(output_path, person_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_path = os.path.join(save_folder, image_filename)
            img.save(save_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

print("Preprocessing complete.")


