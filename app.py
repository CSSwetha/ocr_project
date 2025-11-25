import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import platform
import shutil
import os
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


# ---------- AUTO-DETECT TESSERACT ----------
def configure_tesseract():
    system = platform.system()

    paths = {
        "Linux": ["/usr/bin/tesseract", "/usr/local/bin/tesseract"],
        "Darwin": ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"],
        "Windows": [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
    }

    for p in paths.get(system, []):
        if shutil.which(p) or os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return True

    return False


if not configure_tesseract():
    st.error("❌ Tesseract not detected. Ensure packages.txt contains:\n"
             "`tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-hin` and restart deployment.")
    st.stop()

st.success(f"✔ Tesseract detected: {pytesseract.get_tesseract_version()}")


# ---------- PDF OCR FUNCTION ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    pages = convert_from_path(pdf_path, dpi=300)
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="OCRify LENS", layout="centered")

st.markdown("""
<h1 style="text-align:center">📄 OCRify LENS</h1>
<p style="text-align:center">Extract text, enhance scanned documents & translate!</p>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("📄 Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

doc_text = ""
image_path = ""


# ---------- FILE PROCESSING ----------
if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Performing OCR on PDF..."):
            doc_text = extract_text_from_pdf("temp.pdf")

        first_page = convert_from_path("temp.pdf", dpi=300)[0]
        image_path = "preview.png"
        first_page.save(image_path)

    else:
        image_path = f"input.{ext}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        doc_text = pytesseract.image_to_string(Image.open(image_path))

    st.success("✔ OCR completed!")
    st.image(image_path, caption="📌 Uploaded Preview", width=400)


with st.expander("✨ Advanced Features"):

    # 🔹 Extract Again With Language Selection
    if st.checkbox("📝 Extract Text with Language Selection"):
        langs = {
            "English": "eng", "Hindi": "hin", "Tamil": "tam", "Telugu": "tel",
            "Kannada": "kan", "French": "fra", "German": "deu", "Spanish": "spa"
        }
        selected = st.multiselect("Select Languages", list(langs.keys()), default=["English"])
        code = "+".join([langs[l] for l in selected])

        new_text = pytesseract.image_to_string(Image.open(image_path), lang=code)
        st.text_area("📟 Extracted Text", new_text, height=200)

        if st.checkbox("🌍 Translate to English"):
            detected = detect(new_text)
            translated = GoogleTranslator(source=detected, target="en").translate(new_text)
            st.success(f"Detected language: {detected}")
            st.text_area("📘 Translation", translated, height=200)


    # 🔹 Image Inversion
    if st.checkbox("🌓 Invert Image Colors"):
        img = cv2.imread(image_path)
        inverted = cv2.bitwise_not(img)
        cv2.imwrite("inverted.png", inverted)
        st.image("inverted.png")


    # 🔹 Binarization
    if st.checkbox("🖼 Binarization (Otsu Thresholding)"):
        img = cv2.imread(image_path, 0)
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("binary.png", bin_img)
        st.image("binary.png")


    # 🔹 Deskew
    if st.checkbox("🧭 Deskew Text"):
        img = cv2.imread(image_path)

        def get_skew(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = cv2.findNonZero(thresh)
            angle = cv2.minAreaRect(coords)[-1]
            return -(90 + angle) if angle < -45 else -angle

        angle = get_skew(img)
        rotated = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1),
                                 (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite("deskewed.png", rotated)
        st.image("deskewed.png")


    # 🔤 Font Thickness Adjustment
if st.checkbox("🔤 Font Thickness Adjustment"):
    img = cv2.imread(image_path, 0)

    if img is None:
        st.error("⚠️ Unable to load image for thickness adjustment.")
    else:
        blurred = cv2.medianBlur(img, 3)

        thin = cv2.erode(blurred, np.ones((2,2),np.uint8), iterations=1)
        thick = cv2.dilate(blurred, np.ones((2,2),np.uint8), iterations=1)

        st.image(thin, caption="✏️ Thin Text")
        st.image(thick, caption="🖍️ Thick Text")


    # 🔹 Remove Borders
if st.checkbox("🧹 Remove Borders"):
        img = cv2.imread(image_path)

