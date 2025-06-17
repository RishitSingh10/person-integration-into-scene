from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener
import os

# Register HEIC opener
register_heif_opener()

# Path to the directory containing HEIC images
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')

# Iterate through all HEIC files in the directory
for image_file in Path(RAW_DIR).rglob("*.[hH][eE][iI][Cc]"):
    print(f"Converting: {image_file.name}")
    # Open the HEIC file
    image = Image.open(image_file)
    # Convert and save as JPG
    new_name = f"{image_file.stem}.jpg"
    print(new_name)
    image.convert('RGB').save(os.path.join(RAW_DIR, new_name))
    print(f"Saved as: {new_name}")
