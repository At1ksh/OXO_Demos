def smile_yolo():
    import streamlit as st
    import numpy as np
    import cv2
    import tempfile
    from PIL import Image
    from ultralytics import YOLO
    import tensorflow as tf

    first_model_path = r"E:\\Models\\Smile_Line_Detection\\YOLO\\best.pt"
    second_model_path = r"E:\\Models\\Smile_Line_Light_Pop_Detection\\best.pt"
    keras_model_path = r"E:\\Models\\Smile_Line_Detection\\keras\\SmileLine_SubHSEDetectionV2.keras"
    reference_image_path = "C:/Krishna/Jaguar/Photos/smile_line/UpdatedDataset/AutoBiography/IMG_6778.JPG"

    first_yolo = YOLO(first_model_path)
    second_yolo = YOLO(second_model_path)
    keras_model = tf.keras.models.load_model(keras_model_path)

    st.markdown("<h2 style='color:#2e6c80;'>🪥 Smile Line QC Detection</h2>", unsafe_allow_html=True)
    target_choice = st.selectbox("Choose expected class:", ["AutoBiography", "HSE"])

    col1, col2, col3, col4 = st.columns([1, 3, 3, 2])
    col1.markdown("**🔢 Serial No.**")
    col2.markdown("**📌 Reference Image**")
    col3.markdown("**📷 Uploaded Image**")
    col4.markdown("**✅ Result**")

    col1.write("1")
    col2.image(reference_image_path, caption="Expected Format", use_column_width=True)
    placeholder_uploaded = col3.empty()
    placeholder_result = col4.empty()
    placeholder_result.markdown("<div style='color: gray;'>Pending</div>", unsafe_allow_html=True)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def convert_to_grayscale_rgb(img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return np.stack((gray,) * 3, axis=-1)

    def straighten_crop(image_rgb, obb):
        ordered_pts = order_points(obb.astype("float32"))
        (tl, tr, br, bl) = ordered_pts
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        return cv2.warpPerspective(image_rgb, M, (maxWidth, maxHeight))

    def process_image(img_np_rgb):
        gray_rgb = convert_to_grayscale_rgb(img_np_rgb)
        tmp_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(tmp_path, cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR))

        results1 = first_yolo(tmp_path)[0]
        if not results1.obb or results1.obb.conf is None or len(results1.obb.conf) == 0:
            return None, "❌ Not Ok"

        best_idx = np.argmax(results1.obb.conf.cpu().numpy())
        obb = results1.obb.xyxyxyxy[best_idx].cpu().numpy().astype(int)
        return straighten_crop(gray_rgb, obb), None

    def run_quality_check_with_keras(img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (256, 256))
        input_tensor = np.expand_dims(resized, axis=-1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return keras_model.predict(input_tensor, verbose=0)[0][0]

    def second_model_inference(image_rgb):
        gray_rgb = convert_to_grayscale_rgb(image_rgb)
        temp_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR))
        results2 = second_yolo(temp_path)[0]
        return "AutoBiography Detected" if results2.obb and results2.obb.conf is not None and len(results2.obb.conf) > 0 else "HSE Detected"

    image_file = st.file_uploader("Upload your target image:", type=['jpg', 'jpeg', 'png'])

    if image_file:
        img_rgb = Image.open(image_file).convert('RGB')
        img_np_rgb = np.array(img_rgb)
        placeholder_uploaded.image(img_np_rgb, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Processing image..."):
            cropped, err = process_image(img_np_rgb)
            if err:
                placeholder_result.error(err)
            else:
                quality_pred = run_quality_check_with_keras(cropped)
                if quality_pred > 0.5:
                    placeholder_result.error("❌ Not OK")
                else:
                    class_result = second_model_inference(cropped)
                    if target_choice == class_result.replace(" Detected", ""):
                        placeholder_result.success("✅ OK")
                    else:
                        placeholder_result.error("❌ Not OK")