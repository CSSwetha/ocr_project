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

    # ---------------- Extract Again With Language Selection ----------------
    if st.checkbox("📝 Extract Text with Language Selection", key="lang"):
        langs = {
            "English": "eng", "Hindi": "hin", "Tamil": "tam", "Telugu": "tel",
            "Kannada": "kan", "French": "fra", "German": "deu", "Spanish": "spa"
        }
        selected = st.multiselect("Select Languages", list(langs.keys()), default=["English"])
        code = "+".join([langs[l] for l in selected])

        new_text = pytesseract.image_to_string(Image.open(image_path), lang=code)
        st.text_area("📟 Extracted Text", new_text, height=200)

        if st.checkbox("🌍 Translate to English", key="translate"):
            detected = detect(new_text)
            translated = GoogleTranslator(source=detected, target="en").translate(new_text)
            st.success(f"Detected language: {detected}")
            st.text_area("📘 Translation", translated, height=200)

    # ---------------- Invert Colors ----------------
    if st.checkbox("🌓 Invert Image Colors", key="invert"):
        img = cv2.imread(image_path)
        inverted = cv2.bitwise_not(img)
        cv2.imwrite("inverted.png", inverted)
        st.image("inverted.png", caption="Inverted Image Preview")

    # ---------------- Binarization ----------------
    if st.checkbox("⚫ Binary Thresholding (Otsu)", key="binary"):
        img = cv2.imread(image_path, 0)
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("binary.png", bin_img)
        st.image("binary.png", caption="Binarized Image")

    # ---------------- Deskew ----------------
    if st.checkbox("📐 Deskew Image Alignment", key="deskew"):
        img = cv2.imread(image_path)

        def get_skew(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = cv2.findNonZero(thresh)
            angle = cv2.minAreaRect(coords)[-1]
            return -(90 + angle) if angle < -45 else -angle

        angle = get_skew(img)
        rotated = cv2.warpAffine(img,
            cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1),
            (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite("deskewed.png", rotated)
        st.image("deskewed.png", caption=f"Deskewed (Angle: {round(angle,2)}°)")

    # ---------------- Font Thickness ----------------
    if st.checkbox("🔤 Font Thickness Adjustment", key="thickness"):
        img = cv2.imread(image_path, 0)
        blurred = cv2.medianBlur(img, 3)

        thin = cv2.erode(blurred, np.ones((2,2),np.uint8), iterations=1)
        thick = cv2.dilate(blurred, np.ones((2,2),np.uint8), iterations=1)

        st.image(thin, caption="✏️ Thin Text Result")
        st.image(thick, caption="🖍 Thick Text Result")

    # ---------------- Remove Borders (Repaired Logic) ----------------
    if st.checkbox("🧹 Remove Borders (Lines) from Image"):
        if image_path:
            with st.spinner("🧼 Removing all lines and borders with improved accuracy..."):
                # Load image
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Step 1: Create binary inverted image to enhance contrast
                bin_img = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                cv2.THRESH_BINARY, 15, -2)

                # Step 2: Detect horizontal lines
                horizontal = bin_img.copy()
                cols = horizontal.shape[1]
                horizontal_size = max(1, cols // 20)  # Stronger kernel
                horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
                horizontal = cv2.erode(horizontal, horizontalStructure)
                horizontal = cv2.dilate(horizontal, horizontalStructure)

                # Step 3: Detect vertical lines
                vertical = bin_img.copy()
                rows = vertical.shape[0]
                vertical_size = max(1, rows // 20)  # Stronger kernel
                verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
                vertical = cv2.erode(vertical, verticalStructure)
                vertical = cv2.dilate(vertical, verticalStructure)

                # Step 4: Combine both masks
                mask = cv2.add(horizontal, vertical)

                # Step 5: Dilate mask to ensure even faint lines are caught
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)

                # Step 6: Inpaint to remove lines
                cleaned = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

                # Save and display
                border_removed_path = "border_removed_strict.png"
                cv2.imwrite(border_removed_path, cleaned)

                st.image(border_removed_path, caption="🧼 Cleaned Image - No Borders or Lines", use_container_width=True)

                with open(border_removed_path, "rb") as f:
                    st.download_button("📥 Download Cleaned Image", f, "border_removed_strict.png", mime="image/png")
        else:
            st.warning("⚠️ No image found to remove borders.")
