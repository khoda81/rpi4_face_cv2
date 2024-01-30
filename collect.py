import os
import time
from uuid import uuid4
import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

from config import DATASET_PATH, DISPLAY_RESOLUTION, HAARCASCADE_PATH

# Create the output directory if it doesn't exist
DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Initialize the OpenCV face detector
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

# Initialize the PiCamera
camera = PiCamera()
# camera.resolution = (640, 480)  # Adjust resolution as needed
# camera.framerate = 32

rawCapture = PiRGBArray(camera)

# Warm-up the camera
time.sleep(0.1)

# Input the label for the person whose face is being captured
label = input("Enter the label for the person: ")

# File path
file_path = DATASET_PATH / "names.txt"

# Open the file in append mode ('a+')
with open(file_path, "a+") as names_file:
    # Move the cursor to the beginning of the file for reading
    names_file.seek(0)

    # Read all lines and strip newline characters
    names = [line.strip() for line in names_file.readlines()]

    try:
        # Find the index of the label if it exists
        index = names.index(label)
    except ValueError:
        # Label not found, so add it to the file
        print("New Person! Adding to `names.txt`.")
        names_file.write(label + "\n")
        index = len(names)

# Display the index where the label was found or added
print(f"Index of '{label}': {index}")

# Counter for images captured
image_count = 0

# Start capturing images
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Capture frame-by-frame
    image: np.ndarray = frame.array[:, ::-1].copy()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces and capture images
    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cut out the face
        sample = gray[y : y + h, x : x + w]

        # Display the frame
        cv2.imshow(
            "Face Confirmation (press 'y' to confirm)", sample[::-1].swapaxes(0, 1)
        )

        # Prompt user for confirmation
        key = cv2.waitKey(0) & 0xFF

        # If 'y' is pressed, add the image to the dataset
        if key == ord("y"):
            # Increment image count
            image_count += 1

            # Increment image count
            image_id = uuid4()

            # Save the captured face image with the label
            sample_dir = DATASET_PATH / f"{index}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_path = os.path.join(sample_dir, f"{image_id}.jpg")
            cv2.imwrite(image_path, sample)

            print(f"Image {image_count} added to dataset. ({image_path})")

        # Close the face confirmation window
        cv2.destroyWindow("Face Confirmation (press 'y' to confirm)")

    # Resize the image to fit the LCD screen
    image = cv2.resize(image, DISPLAY_RESOLUTION)[::-1].swapaxes(0, 1)

    # Display the frame
    cv2.imshow("Collecting Dataset", image)

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
camera.close()
cv2.destroyAllWindows()

print(
    f"Dataset collection complete. {image_count} images of person {label} saved in {DATASET_PATH}"
)
