import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import os
import cv2
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import platform

DetectorFactory.seed = 0  # deterministic language detection

# ------------------------------------------
# 🔧 Auto Setup for Tesseract (Linux/Cloud)
# ------------------------------------------
def setup_tesseract():
    system = platform.system()

    if system == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    elif system == "Darwin":   # macOS
        pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

    else:  # Linux (Streamlit cloud)
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
        # Install if missing
        if not os.path.exists("/usr/bin/tesseract"):
            os.system("sudo apt-get update")
            os.system("sudo apt-get install -y tesseract-ocr")
            os.system("sudo apt-get install -y tesseract-ocr-eng tesseract-ocr-hin")

setup_tesseract()

# ------------------------------------------
# Streamlit App UI
# ------------------------------------------
st.set_page_config(page_title="OCRify LENS", layout="centered")

st.markdown("""
<h1 style='text-align:center;'>📄 OCRify LENS</h1>
<p style='text-align:center;'>Upload scanned documents to extract text and enhance images</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload a scanned PDF or image", type=["pdf", "jpg", "jpeg", "png"])

doc_text = ""
image_path = ""

# ------------------------------------------
# OCR Helper for PDF
# ------------------------------------------
def extract_text_from_pdf(path):
    text_output = ""
    pages = convert_from_path(path, dpi=300)
    for page in pages:
        text_output += pytesseract.image_to_string(page)
    return text_output

# ------------------------------------------
# File Processing
# ------------------------------------------
if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext == 'pdf':
        with open("temp_file.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Extracting text from PDF with OCR..."):
            doc_text = extract_text_from_pdf("temp_file.pdf")

        first_page = convert_from_path("temp_file.pdf", dpi=300)[0]
        image_path = "preview.png"
        first_page.save(image_path)

    elif file_ext in ['jpg', 'jpeg', 'png']:
        image_path = "uploaded_image." + file_ext
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Extracting text from image..."):
            img = Image.open(image_path)
            doc_text = pytesseract.image_to_string(img)

    st.success("✅ OCR processing complete!")

    st.image(image_path, caption="🖼️ Uploaded Image", width=400)

# ------------------------------------------
# Feature Options
# ------------------------------------------

with st.expander("✨ Featurization Options"):
    if st.checkbox("📝 Extract Text"):
        if image_path:
            available_langs = {
                "English": "eng", "Hindi": "hin", "Telugu": "tel", "Tamil": "tam",
                "Kannada": "kan", "Gujarati": "guj", "Marathi": "mar", "Punjabi": "pan",
                "Urdu": "urd", "French": "fra", "German": "deu", "Spanish": "spa"
            }
            selected_langs = st.multiselect("🌐 Select OCR languages", list(available_langs.keys()), default=["English"])
            selected_codes = "+".join([available_langs[lang] for lang in selected_langs])

            try:
                img = Image.open(image_path)
                extracted_text = pytesseract.image_to_string(img, lang=selected_codes)
                st.text_area("📟 Extracted Text", extracted_text, height=200)

                if st.checkbox("🌍 Translate to English"):
                    detected_lang = detect(extracted_text)
                    translated = GoogleTranslator(source=detected_lang, target='en').translate(extracted_text)
                    st.text(f"Detected language: {detected_lang}")
                    st.text_area("📘 Translation", translated, height=200)

            except Exception as e:
                st.error(f"❌ OCR Error: {e}")
        else:
            st.warning("⚠️ Upload an image first.")

    if st.checkbox("🌓 Invert Colors"):
        if image_path:
            img = cv2.imread(image_path)
            inverted = cv2.bitwise_not(img)
            path = "inverted.png"
            cv2.imwrite(path, inverted)
            st.image(path)
        else:
            st.warning("⚠️ No image found.")


    if st.checkbox("🖼️ Binarization"):
        if image_path:
            img = cv2.imread(image_path, 0)
            _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            path = "binary.png"
            cv2.imwrite(path, bin_img)
            st.image(path)
        else:
            st.warning("⚠️ Upload an image first.")

st.success("🚀 App Ready!")


