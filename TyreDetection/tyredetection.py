import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from io import BytesIO
import base64

model = YOLO("TyreDetection/my_model.pt")

def tyre_yolo():
    st.markdown("<h2 style='color:#2e6c80;'>🛞 Tyre QC with YOLOv8</h2>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Tyre Images")

    modelselect = st.selectbox('Select Car Model:', ['Autobiography', 'HSE'])

    uploaded_files = st.file_uploader("Upload up to 4 tyre images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) <= 4:
        summary = []

        for idx, uploaded_file in enumerate(uploaded_files, 1):
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner(f"🔍 Processing image {uploaded_file.name}..."):
                results = model(image)
                result_img = results[0].plot()

                boxes = results[0].boxes
                if len(boxes) == 0:
                    summary.append({
                        "serial": idx,
                        "image": result_img,
                        "prediction": "❌ No prediction"
                    })
                else:
                    max_conf = -1
                    max_cls = None
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf > max_conf:
                            max_conf = conf
                            max_cls = cls
                    class_name = results[0].names[max_cls] if max_cls is not None else "Unknown"
                    output_class = "Autobiography Tyre: R22 Michelin Tyre" if class_name != "hse_tyre" else "HSE Tyre: R21 Pirelli Tyre"

                    summary.append({
                        "serial": idx,
                        "image": result_img,
                        "prediction": f"✅ {output_class}"
                    })

        st.markdown("### ✅ Results Table")

        def image_to_html(img):
            buf = BytesIO()
            img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
            img.thumbnail((640, 480))
            img.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/png;base64,{data}" style="display:block;margin:auto;" />'

        df = pd.DataFrame(summary)
        df['image'] = df['image'].apply(image_to_html)
        st.write(df.to_html(escape=False, columns=["serial", "image", "prediction"], index=False), unsafe_allow_html=True)

    elif uploaded_files and len(uploaded_files) > 4:
        st.warning("⚠️ Please upload no more than four images.")