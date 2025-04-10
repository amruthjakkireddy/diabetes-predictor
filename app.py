
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("random_forest_model.pkl", "rb"))

st.title("Diabetes Prediction App")

# Input form
Pregnancies = st.selectbox("Number of Pregnancies", list(range(0, 20)))
Glucose = st.slider("Glucose Level", 0, 200, 120)
BloodPressure = st.slider("Blood Pressure", 0, 122, 70)
SkinThickness = st.slider("Skin Thickness", 0, 100, 20)
Insulin = st.slider("Insulin", 0, 846, 79)
BMI = st.slider("BMI", 0.0, 70.0, 32.0)
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.slider("Age", 1, 120, 33)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)

    st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")

