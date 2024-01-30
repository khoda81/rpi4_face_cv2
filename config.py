# Directory to save the collected images
from pathlib import Path


DATASET_PATH = Path("dataset")
HAARCASCADE_PATH = Path(
    "/home/20mah/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
)
MODEL_PATH = Path("model/trained_model.xml")

# Resolution for the LCD screen
DISPLAY_RESOLUTION = (320, 240)  # Adjust to your LCD screen's resolution
