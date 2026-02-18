import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("traffic_model.pkl")

# Load encoders
encoders = joblib.load("traffic_label_encoders.pkl")
le_weather = encoders["weather_encoder"]
le_day = encoders["day_encoder"]
le_target = encoders["target_encoder"]

st.title("🚦 Traffic Congestion Prediction System")

st.write("Enter traffic details to predict congestion level")

# User Inputs
vehicle_count = st.number_input("Vehicle Count", min_value=0)
hour = st.slider("Hour of Day", 0, 23)
temperature = st.number_input("Temperature (°C)", min_value=0)

weather = st.selectbox("Weather", le_weather.classes_)
day_type = st.selectbox("Day Type", le_day.classes_)

if st.button("Predict"):

    weather_encoded = le_weather.transform([weather])[0]
    day_encoded = le_day.transform([day_type])[0]

    features = np.array([[vehicle_count, hour, temperature, weather_encoded, day_encoded]])

    prediction = model.predict(features)

    result = le_target.inverse_transform(prediction)

    st.success(f"Predicted Congestion Level: {result[0]}")
