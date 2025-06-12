import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import os
from tyredetection import tyre_yolo

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
    st.title("🔍 OXO Checking in CAL Line")

    # Component selection dropdown
    component = st.selectbox(
        "Select the car component to OXO check:",
        ["Smile line", "Interior seats", "Speaker system", "Car lettering", "Bonnet color", "Tyres"]
    )

    if component != "Tyres":
        st.info("OXO checking for this component is being worked on right now.")
    else:
        tyre_yolo()

if __name__ == "__main__":
    main()