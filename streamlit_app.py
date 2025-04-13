# streamlit_app.py
import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and feature names
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("💓 Heart Disease Risk Predictor")

st.markdown("Enter the patient's health information to assess their risk of heart disease.")

# User inputs
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
cp_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert to input dict (one-hot style)
input_dict = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': chol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_M': int(sex == 'M'),
    'ChestPainType_ASY': int(cp_type == 'ASY'),
    'ChestPainType_NAP': int(cp_type == 'NAP'),
    'ChestPainType_TA': int(cp_type == 'TA'),
    'RestingECG_LVH': int(resting_ecg == 'LVH'),
    'RestingECG_ST': int(resting_ecg == 'ST'),
    'ExerciseAngina_Y': int(exercise_angina == 'Y'),
    'ST_Slope_Flat': int(st_slope == 'Flat'),
    'ST_Slope_Up': int(st_slope == 'Up'),
}

# Build full input array (in the same order as training data)
input_array = np.zeros(len(features))
for i, col in enumerate(features):
    input_array[i] = input_dict.get(col, 0)

# Scale the input
input_scaled = scaler.transform([input_array])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.error(f"🔴 High Risk of Heart Disease (Confidence: {prob:.2%})")
    else:
        st.success(f"🟢 Low Risk of Heart Disease (Confidence: {1 - prob:.2%})")
