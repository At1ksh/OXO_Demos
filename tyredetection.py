import streamlit as st
from ultralytics import YOLO
from PIL import Image

def tyre_yolo():
    st.write("Upload up to four photos for YOLO processing. The results will be shown below each image.")
    
    modelselect=st.selectbox('Select car model',['Autobiography','HSE'])

    model = YOLO("my_model.pt")

    uploaded_files = st.file_uploader("Choose up to 4 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) <= 4:
        summary = []
        images = []
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            image = Image.open(uploaded_file)
            with st.spinner(f"Running YOLO on {uploaded_file.name}..."):
                results = model(image)
                result_img = results[0].plot()
                images.append(result_img)

                boxes = results[0].boxes
                if len(boxes) == 0:
                    summary.append({
                        "serial": idx,
                        "image": result_img,
                        "prediction": "No prediction"
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
                    if class_name == "hse_tyre":
                        output_class = "HSE Tyre: R21 Pirelli Tyre"
                    else:
                        output_class = "Autobiography Tyre: R22 Michelin Tyre"

                    summary.append({
                        "serial": idx,
                        "image": image,
                        "prediction": output_class
                    })
        if summary:
            st.markdown("### Results Table")
            import pandas as pd
            from io import BytesIO
            import base64

            def image_to_html(img):
                buf = BytesIO()
                img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
                img.thumbnail((640, 480))
                img.save(buf, format="PNG")
                data = base64.b64encode(buf.getvalue()).decode()
                return f'<img src="data:image/png;base64,{data}" style="display:block;margin:auto;" />'

            df = pd.DataFrame(summary)
            df['image'] = df['image'].apply(image_to_html)
            st.write(
                df.to_html(escape=False, columns=["serial", "image", "prediction"], index=False),
                unsafe_allow_html=True
            )
    elif uploaded_files and len(uploaded_files) > 4:
        st.warning("Please upload no more than four images.")