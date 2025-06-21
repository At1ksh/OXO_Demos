import streamlit as st
import cv2
from PIL import Image
from TyreDetection.tyredetection import tyre_yolo
from FrontDoorFinisherDetection.frontdoorfinisherdetector import fdr_finisher_ocr
from smilelinedetection.smiledetector import smile_yolo
from treadplatedetection.treadplatedetector import treadplate_ocr
from steeringwheeldetection.steeringwheeldetector import steeringwheel_yolo

def main():
    st.set_page_config(page_title="OXO Checking in CAL Line", page_icon="🔍", layout="wide")
    st.markdown(
        """
        <style>
        .main {background-color: #f0f2f6;}
        .stButton>button {background-color: #4CAF50; color: white;}
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center; color: red;'>🔍 OXO Checking in CAL Line</h1>", unsafe_allow_html=True)
    st.markdown("---")

    component = st.selectbox(
        "Select the car component to OXO check:",
        ["Smile line", "Tyres", "Interior seats", "Speaker system", "Car lettering","Treadplate","Front Door Finisher","Steering Wheel"]
    )

    if component == "Tyres":
        tyre_yolo()
    elif component == "Smile line":
        smile_yolo()
    elif component == "Steering Wheel":
        steeringwheel_yolo()
    elif component == "Treadplate":
        treadplate_ocr()
    elif component == "Front Door Finisher":
        fdr_finisher_ocr()
    else:
        st.info("OXO checking for this component is being worked on right now.")

if __name__ == "__main__":
    main()
