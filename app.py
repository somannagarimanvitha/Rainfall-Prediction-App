import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load Models
rf_model = pickle.load(open("models/rf_model.pkl", "rb"))
xgb_model = pickle.load(open("models/xgb_model.pkl", "rb"))

st.title("🌧️ Rainfall Prediction System")
st.write("Predict rainfall using Machine Learning Models")

st.sidebar.header("Enter Weather Parameters")

def user_input():
    pressure = st.sidebar.number_input("Pressure", 900,1100,1000)
    humidity = st.sidebar.slider("Humidity",0,100,60)
    dewpoint = st.sidebar.number_input("Dew Point",0,50,25)
    winddirection = st.sidebar.number_input("Wind Direction",0,360,180)
    windspeed = st.sidebar.number_input("Wind Speed",0,100,20)
    cloud = st.sidebar.slider("Cloud Cover",0,100,50)
    sunshine = st.sidebar.number_input("Sunshine",0,15,7)

    data = {
        'pressure':pressure,
        'humidity':humidity,
        'dewpoint':dewpoint,
        'winddirection':winddirection,
        'windspeed':windspeed,
        'cloud':cloud,
        'sunshine':sunshine
    }

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("User Input Data")
st.write(input_df)

# Predictions
rf_pred = rf_model.predict(input_df)
xgb_pred = xgb_model.predict(input_df)

st.subheader("Prediction Results")

st.write("Random Forest Prediction:",
         "🌧️ Rainfall Expected" if rf_pred[0]==1 else "☀️ No Rainfall")

st.write("XGBoost Prediction:",
         "🌧️ Rainfall Expected" if xgb_pred[0]==1 else "☀️ No Rainfall")

# Accuracy Section
st.subheader("Model Accuracy")
st.success("Random Forest Accuracy: 0.77")
st.success("XGBoost Accuracy: 0.79 (Best Model)")

# Visualization Section
st.subheader("Dataset Visualizations")

data = pd.read_csv("Rainfall.csv")

fig, ax = plt.subplots()
sns.countplot(x="rainfall", data=data, ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, ax=ax2)
st.pyplot(fig2)