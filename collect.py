import os
from pathlib import Path
import time
from uuid import uuid4
import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

from config import DATASET_PATH, HAARCASCADE_PATH

# Create the output directory if it doesn't exist
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Initialize the OpenCV face detector
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

# Initialize the PiCamera
camera = PiCamera()

# camera.resolution = (640, 480)  # Adjust resolution as needed
# camera.framerate = 32
rawCapture = PiRGBArray(camera)

# Warm-up the camera
time.sleep(0.1)

# Counter for images captured
image_count = 0

# Input the label for the person whose face is being captured
label = input("Enter the label for the person: ")

# TODO use "names.txt"
assert "TODO"

# Start capturing images
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Capture frame-by-frame
    image: np.ndarray = frame.array.copy()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces and capture images
    for x, y, w, h in faces:
        # TODO Ask for user confirmation before adding each image

        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Increment image count
        image_id = uuid4()

        # Save the captured face image with the label
        image_path = os.path.join(DATASET_PATH, f"{label}/{image_id}.jpg")
        cv2.imwrite(image_path, gray[y : y + h, x : x + w])

        # Display a message indicating the image capture
        # Calculate text size
        text = f"{label}?\nPress Enter to add."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Calculate text position to center it under the box
        text_x = x + (w - text_size[0]) // 2
        text_y = y + h + text_size[1] + 5  # Adjust 5 pixels below the box

        # Draw text
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness=2,
        )

        # Display the frame
        cv2.imshow("Collecting Dataset", image)

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
