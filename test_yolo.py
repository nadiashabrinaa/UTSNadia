import streamlit as st
try:
    from ultralytics import YOLO
    st.success("YOLO terinstal!")
except ImportError:
    st.error("YOLO belum terinstal!")
