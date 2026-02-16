import streamlit as st
import joblib
import numpy as np
import os

st.title("🎓 Student Performance Prediction App")

model_path = "C:/Users/user/student-performance-app/student_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found!")
    st.stop()

st.sidebar.header("Enter Student Data")

class_participation = st.sidebar.number_input("Class Participation", 0, 100)
weekly_self_study_hours = st.sidebar.number_input("Weekly Self Study Hours", 0, 50)
attendance_percentage = st.sidebar.number_input("Attendance Percentage", 0, 100)

if st.button("Predict"):

    input_data = np.array([[class_participation,
                            weekly_self_study_hours,
                            attendance_percentage]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Performance: {prediction[0]:.2f}")
