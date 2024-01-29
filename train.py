import cv2
import os
import numpy as np

from config import DATASET_PATH, MODEL_PATH


# Initialize lists to store faces and labels
faces = []
labels = []

with open(DATASET_PATH / "names.txt") as names_file:
    names = list(names_file.readline())

# Loop through each subdirectory in the dataset directory
for subdir in os.listdir(DATASET_PATH):
    subdir_path = os.path.join(DATASET_PATH, subdir)

    # Skip if it's not a directory
    if not os.path.isdir(subdir_path):
        continue

    # Get the label (person's name) from the directory name
    label = int(subdir)
    print(f"Processing images for label: {names[label]!r} ({label=})")

    # Loop through each image file in the subdirectory
    for filename in os.listdir(subdir_path):
        # Read the image in grayscale
        img_path = os.path.join(subdir_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image if necessary
        # img = cv2.resize(img, (100, 100))

        # Append the face and label to the lists
        faces.append(img)
        labels.append(label)

# Convert the lists to NumPy arrays
faces = np.array(faces)
labels = np.array(labels)

# Create LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the face recognizer
face_recognizer.train(faces, labels)

# Save the trained model
face_recognizer.save(MODEL_PATH)

print("Training completed and model saved.")
