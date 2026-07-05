import streamlit as st
import joblib
import os
import numpy as np

# ---------- Load model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_sales_model.pkl")

model = joblib.load(MODEL_PATH)

# ---------- UI ----------
st.title("📊 Sales Forecasting Dashboard")

st.write("Enter values to predict sales:")

# Example inputs (change based on your model features)
store = st.number_input("Store ID", min_value=1)
item = st.number_input("Item ID", min_value=1)
day = st.number_input("Day", min_value=1, max_value=31)
month = st.number_input("Month", min_value=1, max_value=12)
year = st.number_input("Year", min_value=2000, max_value=2100)

# ---------- Prediction ----------
if st.button("Predict Sales"):
    input_data = np.array([[store, item, day, month, year]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: {prediction[0]}")
