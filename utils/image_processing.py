import pytesseract
from PIL import Image

def process_image(image_path):
    """
    Extracts text from an image using OCR (Tesseract).
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""
