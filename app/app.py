import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict

st.title("AI Fake News Detector")

user_input = st.text_area("Enter news text to verify:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        result = predict(user_input)
        st.subheader("Prediction")
        st.write("**Result:**", result["prediction"])
        st.write("**Confidence:**", round(result["confidence"], 3))