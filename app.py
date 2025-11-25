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
import shutil

DetectorFactory.seed = 0

# --------- TESSERACT AUTO-CONFIGURATION ---------
def configure_tesseract():
    system = platform.system()

    if system == "Linux":
        candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
    elif system == "Darwin":
        candidates = ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]
    else:  # Windows
        candidates = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
        ]

    for path in candidates:
        if shutil.which(path) or os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True

    return False

# Run setup
if not configure_tesseract():
    st.error("❌ Tesseract OCR is not installed or not detected. Ensure installation or include in packages.txt for deployment.")
    st.stop()
else:
    st.success(f"Tesseract Detected ✔ Version: {pytesseract.get_tesseract_version()}")

# -------- UI CONFIG --------
st.set_page_config(page_title="OCRify LENS", layout="centered")

st.markdown("""
<h1 style='text-align:center;'>📄 OCRify LENS</h1>
<p style='text-align:center;'>Upload scanned documents to extract text and enhance images</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
doc_text = ""
image_path = ""

# -------- PDF TEXT EXTRACT --------
def extract_pdf_text(pdf_path):
    text = ""
    pages = convert_from_path(pdf_path, dpi=300)
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text

# -------- FILE HANDLING --------
if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()

    if ext == 'pdf':
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("🔍 Extracting PDF text..."):
            doc_text = extract_pdf_text("temp.pdf")
        preview = convert_from_path("temp.pdf", dpi=300)[0]
        image_path = "preview.png"
        preview.save(image_path)

    else:
        image_path = f"input.{ext}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("🔍 Extracting text from image..."):
            doc_text = pytesseract.image_to_string(Image.open(image_path))

    st.success("✅ OCR Completed!")
    st.image(image_path, caption="🖼 Uploaded File Preview", width=400)

# -------- OPTIONS PANEL --------
with st.expander("✨ More Options"):

    if st.checkbox("📝 Extract OCR Text Again"):
        langs = {
            "English": "eng", "Hindi": "hin", "Tamil": "tam", "Telugu": "tel",
            "Kannada": "kan", "French": "fra", "German": "deu", "Spanish": "spa"
        }
        selected = st.multiselect("🌐 Choose Language(s)", list(langs.keys()), default=["English"])
        lang_code = "+".join([langs[l] for l in selected])

        try:
            new_text = pytesseract.image_to_string(Image.open(image_path), lang=lang_code)
            st.text_area("📟 Extracted Text", new_text, height=200)

            if st.checkbox("🌍 Translate to English"):
                detected = detect(new_text)
                translated = GoogleTranslator(source=detected, target="en").translate(new_text)
                st.text_area("📘 Translation", translated, height=200)
        except Exception as e:
            st.error(f"❌ OCR Failed: {e}")

    if st.checkbox("🌓 Invert Colors"):
        img = cv2.imread(image_path)
        inv = cv2.bitwise_not(img)
        path = "inverted.png"
        cv2.imwrite(path, inv)
        st.image(path)

    if st.checkbox("🖼 Binarize Image"):
        img = cv2.imread(image_path, 0)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        path = "binary.png"
        cv2.imwrite(path, binary)
        st.image(path)

st.success("🚀 App Ready and Working ✔")
