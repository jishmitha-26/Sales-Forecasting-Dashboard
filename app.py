import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------- PATHS -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "xgboost_sales_model.pkl")

# Optional dataset (only if exists)
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# ----------------- LOAD MODEL -----------------
model = joblib.load(MODEL_PATH)

# ----------------- LOAD DATA (SAFE) -----------------
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = None

# ----------------- UI -----------------
st.title("📊 Sales Forecasting Dashboard")

# ----------------- SIDEBAR -----------------
st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Analysis", "Prediction"]
)

# ----------------- OVERVIEW -----------------
if option == "Overview":
    st.subheader("📌 Project Overview")
    st.write("This dashboard predicts sales using Machine Learning (XGBoost model).")

    if df is not None:
        st.success("Dataset loaded successfully")
        st.write("Shape:", df.shape)
    else:
        st.warning("Dataset not found. Only prediction mode available.")

# ----------------- DATA ANALYSIS -----------------
elif option == "Data Analysis":
    st.subheader("📈 Data Analysis")

    if df is not None:
        st.write(df.head())

        if "sales" in df.columns:
            st.line_chart(df["sales"])
        else:
            st.info("No 'sales' column found in dataset.")
    else:
        st.error("Dataset not available")

# ----------------- PREDICTION -----------------
elif option == "Prediction":
    st.subheader("🤖 Predict Sales")

    store = st.number_input("Store ID", min_value=1)
    item = st.number_input("Item ID", min_value=1)
    day = st.number_input("Day", 1, 31)
    month = st.number_input("Month", 1, 12)
    year = st.number_input("Year", 2000, 2100)

    if st.button("Predict"):
        input_data = np.array([[store, item, day, month, year]])
        prediction = model.predict(input_data)

        st.success(f"📦 Predicted Sales: {prediction[0]}")
