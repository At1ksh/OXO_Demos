def steeringwheel_yolo():
    import streamlit as st
    import numpy as np
    import cv2
    import tempfile
    from PIL import Image
    from ultralytics import YOLO

    # === Define your paths ===
    first_yolo_path = r"E:\Models\YoloSportVariantDetection\best.pt"  # Defect detection
    second_yolo_path = r"E:\ModelBuilding\YOLOforSteering\YOLOFull\runs\obb\yolov8-obb-run\weights\best.pt"  # Material detection
    reference_image_path = r"C:\Krishna\Jaguar\Steering\Quality_of_material_around_the_Wheel\Dataset\TestingDataset\ActualImages\2.jpg" # <-- Replace with your reference

    first_model = YOLO(first_yolo_path)
    second_model = YOLO(second_yolo_path)

    st.markdown("<h2 style='color:#2e6c80;'>🛠️ Steering Wheel QC Detection</h2>", unsafe_allow_html=True)

    variant = st.selectbox("Choose expected variant:", ["Autobiography", "HSE"])

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

    def detect_objects(model, image_pil):
    # Ensure it’s RGB and converted to something YOLOv8 accepts
        if not isinstance(image_pil, Image.Image):
            image_pil = Image.open(image_pil).convert('RGB')

        results = model.predict(image_pil, task="obb")  # PIL is okay
        boxes = results[0].obb
        class_names = results[0].names

        if boxes is None:
            return [], results

        results_list = []
        for i in range(len(boxes.data)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            name = class_names[cls_id]
            results_list.append({"name": name, "confidence": conf})
        return results_list, results


    def extract_color(material_name):
        if material_name and '_' in material_name:
            return '_'.join(material_name.split('_')[:-1])  # supports multi-word like "Soft_Leather"
        return None

    image_file = st.file_uploader("Upload your target image:", type=['jpg', 'jpeg', 'png'])

    if image_file:
        img = Image.open(image_file).convert('RGB')
        img_rgb = Image.open(image_file).convert('RGB')
        img_np = np.array(img_rgb)
        placeholder_uploaded.image(img_np, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Running detections..."):
            # Run first model
            first_results, first_full = detect_objects(first_model, img)
            #st.image(first_full[0].plot(), caption="First Model Inference", use_column_width=True)

            if len(first_results) > 0:
                placeholder_result.error("❌ Not OK")
            else:
                # Run second model
                second_results, second_full = detect_objects(second_model, img)
                #st.image(second_full[0].plot(), caption="Second Model Inference", use_column_width=True)

                # Class filters
                dab_classes = ["Caraway_Dab", "Garnet_Dab", "Perlion_Dab"]
                ip_classes = ["Caraway_IP", "Garnet_IP", "Perlion_IP"]
                steer_classes = ["Soft Leather Steering", "Wooden Steering"]

                def get_best_match(classes):
                    subset = [res for res in second_results if res['name'] in classes]
                    if not subset:
                        return None
                    return sorted(subset, key=lambda x: x['confidence'], reverse=True)[0]['name']

                dab = get_best_match(dab_classes)
                ip = get_best_match(ip_classes)
                steering = get_best_match(steer_classes)

                dab_color = extract_color(dab)
                ip_color = extract_color(ip)

                # Logic 1: Dab and IP color match
                if dab and ip and dab_color != ip_color:
                    placeholder_result.error("❌ Not OK")
                else:
                    # Logic 2: Variant and Steering match
                    if variant == "Autobiography":
                        if steering == "Wooden Steering":
                            placeholder_result.success("✅ OK")
                        else:
                            placeholder_result.error("❌ Not OK")
                    elif variant == "HSE":
                        if steering == "Soft Leather Steering":
                            placeholder_result.success("✅ OK")
                        else:
                            placeholder_result.error("❌ Not OK")
