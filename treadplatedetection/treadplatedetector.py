import streamlit as st
import easyocr
import cv2
from PIL import Image
import numpy as np
import os
from fuzzywuzzy import fuzz
import tempfile


def treadplate_ocr():
    st.markdown("<h2 style='color:#2e6c80;'>🛡️ Treadplate Text Detection</h2>", unsafe_allow_html=True)
    target_choice = st.selectbox("Choose expected class:", ["AutoBiography", "HSE"])

    # === Column Setup ===
    col1, col2, col3, col4 = st.columns([1, 3, 3, 2])
    col1.markdown("**🔢 Serial No.**")
    col2.markdown("**📌 Reference Image**")
    col3.markdown("**📷 Uploaded Image**")
    col4.markdown("**✅ Result**")

    col1.write("1")
    ref_image_path = (
        r"C:\Krishna\Jaguar\Writteninterior\5.jpg"
        if target_choice == "AutoBiography"
        else r"C:\Krishna\Jaguar\Writteninterior\1_landscape.jpg"
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
            score_hse = fuzz.ratio(combined_text, "Range Rover")
            score = max(score_auto, score_hse)

            if score > best_score:
                best_score = score
                best_angle = angle
                best_text = combined_text

        best_rotated = rotate_image(image, best_angle)
        return best_rotated, best_angle, best_text


    def straighten_image_if_portrait(image_path):
        with Image.open(image_path) as img:
            rotated_by_exif = False

            # Try EXIF auto-orientation
            try:
                exif = img._getexif()
                if exif is not None:
                    orientation = exif.get(274)
                    if orientation == 3:
                        st.write("1")
                        img = img.rotate(180, expand=True)
                        rotated_by_exif = True
                    elif orientation == 6:
                        st.write("2")
                        img = img.rotate(-90, expand=True)
                        rotated_by_exif = True
                    elif orientation == 8:
                        st.write("3")
                        img = img.rotate(90, expand=True)
                        rotated_by_exif = True
            except Exception:
                pass  # No EXIF or not accessible

            # Check size after EXIF correction
            width, height = img.size

            # Only rotate if still portrait
            if height > width and not rotated_by_exif:
                img = img.rotate(-90, expand=True)

            # Save
            file_root, file_ext = os.path.splitext(image_path)
            if file_ext.lower() not in [".jpg", ".jpeg", ".png"]:
                file_ext = ".jpg"
            rotated_path = file_root + "_rotated" + file_ext
            img.save(rotated_path)

            return rotated_path, (height > width)


            

    image_file = st.file_uploader("Upload your target image:", type=['jpg', 'jpeg', 'png'])

    if image_file:
        # Save temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(image_file.read())
        temp_input_path = temp_file.name

        img_rgb = Image.open(temp_input_path).convert('RGB')
        img_np_rgb = np.array(img_rgb)
        

        with st.spinner("🔍 Processing image..."):
            # Step 1: Auto-orient using OCR
            straightened_image, best_angle, best_text = get_best_orientation(temp_input_path)

            # Step 2: Convert to RGB for display
            straightened_rgb = cv2.cvtColor(straightened_image, cv2.COLOR_BGR2RGB)

            # Step 3: OCR again (if you want bounding boxes)
            reader = easyocr.Reader(['en'])
            results = reader.readtext(straightened_image)

            # Step 4: Draw bounding boxes
            for (bbox, text, prob) in results:
                (top_left, _, bottom_right, _) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(straightened_rgb, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(straightened_rgb, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Step 5: Combine text and display
            combined_text = " ".join([text for (_, text, _) in results])
            placeholder_uploaded.image(straightened_rgb, caption="Uploaded Image", use_column_width=True)

            # Step 6: Fuzzy match
            score_auto = fuzz.ratio(combined_text, "AutoBiography")
            score_hse = fuzz.ratio(combined_text, "Range Rover")

            if target_choice == "AutoBiography":
                final_result = "✅ OK" if score_auto > score_hse else "❌ Not OK"
            else:  # HSE
                final_result = "✅ OK" if score_hse > score_auto else "❌ Not OK"

            if "OK" in final_result:
                placeholder_result.success(final_result)
            else:
                placeholder_result.error(final_result)
