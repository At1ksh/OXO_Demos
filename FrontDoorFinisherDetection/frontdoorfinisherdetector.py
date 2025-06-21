import streamlit as st
import easyocr
import cv2
from PIL import Image
import numpy as np
import os
from fuzzywuzzy import fuzz
import tempfile


def fdr_finisher_ocr():
    st.markdown("<h2 style='color:#2e6c80;'>🛡️ Front Door Finisher Detection</h2>", unsafe_allow_html=True)
    target_choice = st.selectbox("Choose expected class:", ["AutoBiography", "HSE"])

    # === Column Setup ===
    col1, col2, col3, col4 = st.columns([1, 3, 3, 2])
    col1.markdown("**🔢 Serial No.**")
    col2.markdown("**📌 Reference Image**")
    col3.markdown("**📷 Uploaded Image**")
    col4.markdown("**✅ Result**")

    col1.write("1")
    ref_image_path = (
        r"C:\Krishna\Jaguar\FrontDoorFinisher\4.jpg"
        if target_choice == "AutoBiography"
        else r"C:\Krishna\Jaguar\FrontDoorFinisher\2.jpg"
    )
    col2.image(ref_image_path, caption="Expected Format", use_column_width=True)

    placeholder_uploaded = col3.empty()
    placeholder_result = col4.empty()
    placeholder_result.markdown("<div style='color: gray;'>Pending</div>", unsafe_allow_html=True)

    reader = easyocr.Reader(['en'])

    def rotate_image(image, angle):
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def get_best_orientation(image_path):
        image = cv2.imread(image_path)
        orientations = [0, 90, 180, 270]

        best_score = -1
        best_angle = 0
        best_text = ""

        for angle in orientations:
            rotated = rotate_image(image, angle)
            result = reader.readtext(rotated)
            combined_text = " ".join([text for (_, text, _) in result])

            # Fuzzy match to both targets
            score_auto = fuzz.ratio(combined_text, "AutoBiography")
            score = score_auto

            if score > best_score:
                best_score = score
                best_angle = angle
                best_text = combined_text

        best_rotated = rotate_image(image, best_angle)
        return best_rotated

    image_file = st.file_uploader("Upload your target image:", type=['jpg', 'jpeg', 'png'])


    if image_file:
        # Save temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(image_file.read())
        temp_input_path = temp_file.name
        

        with st.spinner("🔍 Processing image..."):
            # Step 1: Auto-orient using OCR
            straightened_image = get_best_orientation(temp_input_path)

            # Step 2: Convert to RGB for display
            straightened_rgb = cv2.cvtColor(straightened_image, cv2.COLOR_BGR2RGB)
            placeholder_uploaded.image(straightened_rgb, caption="Straightened Image", use_column_width=True)

            # Step 3: OCR again (if you want bounding boxes)
            reader = easyocr.Reader(['en'])
            results = reader.readtext(straightened_image)
            
            
           # st.write(results)
            
            if len(results) > 0:
                final_result="✅ OK" if target_choice=="AutoBiography" else "❌ Not OK"
            else:
                final_result = "✅ OK" if target_choice == "HSE" else "❌ Not OK"
            
            if "OK" in final_result:
                placeholder_result.success(final_result)
            else:
                placeholder_result.error(final_result)

                
