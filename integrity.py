from PIL import Image
import os

def check_image_integrity(image_path):
    try:
        Image.open(image_path).verify()
        return True  # Image is intact
    except Exception as e:
        print(f"Image {image_path} is corrupted: {e}")
        return False  # Image is corrupted

# Specify the path to the downloaded image
image_path = "/Users/arshiailaty/Documents/Ashter/dog_waste_images/dog waste_1.jpg"

if os.path.exists(image_path):
    if check_image_integrity(image_path):
        print("Image is intact and can be opened.")
    else:
        print("Image is corrupted or damaged.")
else:
    print("Image file not found.")
