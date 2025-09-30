import pytesseract
from pdf2image import convert_from_path

# Set paths correctly
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(
        pdf_path,
        poppler_path=r"C:\Users\sreni\OneDrive\Desktop\ocr\poppler-24.08.0\Library\bin"
    )
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text
