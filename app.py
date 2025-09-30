import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # deterministic language detection

st.set_page_config(page_title="OCRify LENS", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

st.markdown(
    """
    <h1 style='text-align: center;'>📄 OCRify LENS</h1>
    <p style='text-align: center;'>Upload scanned documents to extract text and enhance images</p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📄 Upload a scanned PDF or image", type=["pdf", "jpg", "jpeg", "png"])

doc_text = ""
image_path = ""
uploaded_image_previewed = False  # Track if we showed preview

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext == 'pdf':
        with open("temp_file.pdf", "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("🔍 Extracting text from PDF with OCR..."):
            doc_text = extract_text_from_pdf("temp_file.pdf")

        images = convert_from_path("temp_file.pdf", dpi=300)
        first_page = images[0]
        image_path = "first_page.png"
        first_page.save(image_path)

    elif file_ext in ['jpg', 'jpeg', 'png']:
        image_path = f"temp_image.{file_ext}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Extracting text from image with OCR..."):
            image = Image.open(image_path)
            doc_text = pytesseract.image_to_string(image)

    st.success("✅ Text extracted successfully!")

    # ✅ Show Uploaded Image Preview
    if image_path:
        st.image(image_path, caption="🖼️ Uploaded Image Preview", width=400)
        uploaded_image_previewed = True


with st.expander("✨ Featurization Options"):

    if st.checkbox("📝 Extract Text from Image"):
        if image_path:
            available_langs = {
                "English": "eng", "Hindi": "hin", "Telugu": "tel", "Tamil": "tam",
                "Kannada": "kan", "Gujarati": "guj", "Marathi": "mar", "Punjabi": "pan",
                "Urdu": "urd", "French": "fra", "German": "deu", "Spanish": "spa"
            }
            selected_langs = st.multiselect("🌐 Select OCR language(s) to use", list(available_langs.keys()), default=["English"])
            selected_codes = "+".join([available_langs[lang] for lang in selected_langs])

            with st.spinner(f"🔍 Extracting text using languages: {selected_codes}"):
                try:
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    image = Image.open(image_path)
                    extracted_text = pytesseract.image_to_string(image, lang=selected_codes)
                    st.text_area("📟 Extracted Text", value=extracted_text, height=200)
                except pytesseract.TesseractError as e:
                    st.error(f"❌ OCR failed: {e}")
                    extracted_text = ""

            if extracted_text and st.checkbox("🌍 Translate extracted text to English"):
                with st.spinner("🔤 Detecting language and translating..."):
                    try:
                        detected_lang = detect(extracted_text)
                        translated = GoogleTranslator(source=detected_lang, target='en').translate(extracted_text)
                        st.text(f"🔎 Detected language: {detected_lang}")
                        st.text_area("📘 Translated to English", value=translated, height=200)
                        if st.button("🔊 Convert Translated Text to Speech"):
                            try:
                                engine = pyttsx3.init()
                                engine.setProperty('rate', 150)  # Optional: Adjust speed
                                engine.say(translated)
                                engine.runAndWait()
                                st.success("✅ Voice playback completed!")
                            except Exception as e:
                                st.error(f"❌ Text-to-speech failed: {e}")

                    except Exception as e:
                        st.error(f"❌ Translation failed: {e}")
        else:
            st.warning("⚠️ No image found to extract text.")

    if st.checkbox("🌓 Invert Image Colors"):
        if image_path:
            with st.spinner("🎨 Inverting image colors..."):
                image = cv2.imread(image_path)
                inverted_image = cv2.bitwise_not(image)
                inverted_path = "inverted_image.png"
                cv2.imwrite(inverted_path, inverted_image)

                st.image(inverted_path, caption="🖼️ Inverted Image", use_container_width=True)

                with open(inverted_path, "rb") as f:
                    st.download_button("📥 Download Inverted Image", f, "inverted_image.png", mime="image/png")
        else:
            st.warning("⚠️ No image found to invert.")

    if st.checkbox("🖼️ Binarization (Otsu's Thresholding)"):
        if image_path:
            with st.spinner("🎛️ Applying binarization..."):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                apply_noise_removal = st.checkbox("🔧 Apply noise removal", value=True)

                if apply_noise_removal:
                    kernel = np.ones((3, 3), np.uint8)
                    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
                else:
                    processed_image = binary_image

                binarized_path = "binarized_output.png"
                cv2.imwrite(binarized_path, processed_image)

                st.image(binarized_path, caption="🖼️ Binarized Image" + (" + Denoised" if apply_noise_removal else ""), use_container_width=True)

                with open(binarized_path, "rb") as f:
                    st.download_button("📥 Download Binarized Image", f, "binarized_output.png", mime="image/png")
        else:
            st.warning("⚠️ No image found to binarize.")

    if st.checkbox("🧭 Rotate / Deskew Text Alignment"):
        if image_path:
            with st.spinner("🌀 Rotating and aligning text..."):

                def getSkewAngle(cvImage) -> float:
                    newImage = cvImage.copy()
                    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (9, 9), 0)
                    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
                    dilate = cv2.dilate(thresh, kernel, iterations=2)
                    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    if not contours:
                        return 0
                    largestContour = contours[0]
                    minAreaRect = cv2.minAreaRect(largestContour)
                    angle = minAreaRect[-1]
                    return -1.0 * (90 + angle) if angle < -45 else -1.0 * angle

                def rotateImage(cvImage, angle: float):
                    (h, w) = cvImage.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    return cv2.warpAffine(cvImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                def deskew(cvImage):
                    angle = getSkewAngle(cvImage)
                    return rotateImage(cvImage, -angle)

                image = cv2.imread(image_path)
                deskewed_image = deskew(image)

                deskewed_path = "deskewed_image.png"
                cv2.imwrite(deskewed_path, deskewed_image)

                st.image(deskewed_path, caption="🖼️ Deskewed Image", use_container_width=True)

                with open(deskewed_path, "rb") as f:
                    st.download_button("📥 Download Deskewed Image", f, "deskewed_image.png", mime="image/png")
        else:
            st.warning("⚠️ No image found to rotate/deskew.")

    # Font Thickness Adjustment: Thin or Thick Text
    if st.checkbox("🔤 Font Thickness Adjustment"):
        if image_path:
            col1, col2 = st.columns(2)

            def thin_font(image):
                image = cv2.bitwise_not(image)
                kernel = np.ones((2, 2), np.uint8)
                image = cv2.erode(image, kernel, iterations=1)
                image = cv2.bitwise_not(image)
                return image

            def thick_font(image):
                image = cv2.bitwise_not(image)
                kernel = np.ones((2, 2), np.uint8)
                image = cv2.dilate(image, kernel, iterations=1)
                image = cv2.bitwise_not(image)
                return image

            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            no_noise = cv2.medianBlur(original, 3)

            # Thin font (Erosion)
            eroded_image = thin_font(no_noise)
            eroded_path = "eroded_image.jpg"
            cv2.imwrite(eroded_path, eroded_image)
            with col1:
                st.image(eroded_path, caption="✏️ Thin Font (Erosion)", use_column_width=True)
                with open(eroded_path, "rb") as f:
                    st.download_button("📥 Download Thin Font", f, "thin_font.jpg", "image/jpeg")

            # Thick font (Dilation)
            dilated_image = thick_font(no_noise)
            dilated_path = "dilated_image.jpg"
            cv2.imwrite(dilated_path, dilated_image)
            with col2:
                st.image(dilated_path, caption="🖍️ Thick Font (Dilation)", use_column_width=True)
                with open(dilated_path, "rb") as f:
                    st.download_button("📥 Download Thick Font", f, "thick_font.jpg", "image/jpeg")
        else:
            st.warning("⚠️ Please upload an image first to apply font thickness adjustments.")

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

