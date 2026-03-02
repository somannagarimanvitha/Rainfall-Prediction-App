import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load Trained Models
# -------------------------------
rf_model = pickle.load(open("random_forest_model.pkl", "rb"))
xgb_model = pickle.load(open("xgboost_model.pkl", "rb"))

st.title("🌧 Rainfall Prediction System")

st.write("Enter Weather Details to Predict Rainfall")

# -------------------------------
# User Inputs (ALL FEATURES USED IN TRAINING)
# -------------------------------

temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=70.0)
pressure = st.number_input("Pressure (hPa)", value=1012.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
dewpoint = st.number_input("Dew Point (°C)", value=20.0)
cloud = st.number_input("Cloud Cover (%)", value=50.0)
sunshine = st.number_input("Sunshine (hours)", value=6.0)

# -------------------------------
# Create DataFrame (IMPORTANT FIX)
# -------------------------------
input_data = pd.DataFrame({
    'temperature': [temperature],
    'humidity': [humidity],
    'pressure': [pressure],
    'wind_speed': [wind_speed],
    'dewpoint': [dewpoint],
    'cloud': [cloud],
    'sunshine': [sunshine]
})

st.write("### Input Data")
st.write(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Rainfall"):

    rf_prediction = rf_model.predict(input_data)[0]
    xgb_prediction = xgb_model.predict(input_data)[0]

    st.subheader("Prediction Results")

    if rf_prediction == 1:
        st.success("🌧 Random Forest: Rainfall Expected")
    else:
        st.warning("☀ Random Forest: No Rainfall")

    if xgb_prediction == 1:
        st.success("🌧 XGBoost: Rainfall Expected")
    else:
        st.warning("☀ XGBoost: No Rainfall")
