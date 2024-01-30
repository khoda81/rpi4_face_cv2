import time
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

from config import DATASET_PATH, DISPLAY_RESOLUTION, HAARCASCADE_PATH, MODEL_PATH

# Load the trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read(MODEL_PATH)

# Load the label names from names.txt
with open(DATASET_PATH / "names.txt", "r") as names_file:
    names = [line.strip() for line in names_file.readlines()]

# Initialize the PiCamera
camera = PiCamera()
camera.resolution = (640, 480)  # Adjust resolution as needed
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Warm-up the camera
time.sleep(0.1)

# Initialize the OpenCV face detector
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Convert the frame to a NumPy array
    image = frame.array[:, ::-1].copy()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Recognize faces
    for x, y, w, h in faces:
        # Extract the ROI (Region of Interest) for face recognition
        roi_gray = gray[y : y + h, x : x + w]

        # Recognize the face using the trained model
        label, confidence = face_recognizer.predict(roi_gray)

        # Draw a rectangle around the face
        color = (0, 255, 0)  # Green color
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Display the recognized name and confidence level
        if confidence < 100:
            name = names[label]
            text = f"{name} ({confidence:.2f}%)"
        else:
            text = "Unknown"

        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Resize the image to fit the LCD screen
    image = cv2.resize(image, DISPLAY_RESOLUTION)[::-1].swapaxes(0, 1)

    # Display the frame
    cv2.imshow("Face Recognition", image)

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
camera.close()
cv2.destroyAllWindows()
