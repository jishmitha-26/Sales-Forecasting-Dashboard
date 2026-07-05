import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_sales_model.pkl")

DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# ---------------- LOAD MODEL ----------------
model = None
try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("❌ Model not found or failed to load")

# ---------------- LOAD DATA ----------------
df = None
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
    except:
        df = None

# ---------------- UI ----------------
st.title("📊 Sales Forecasting Dashboard")

# ---------------- SIDEBAR NAVIGATION ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Analytics", "Prediction", "About"]
)

# ================= OVERVIEW =================
if menu == "Overview":
    st.subheader("📌 Project Overview")

    st.write("""
    This dashboard is built for Sales Forecasting using Machine Learning (XGBoost Model).
    
    Features:
    - Data Analysis
    - Sales Visualization
    - Prediction System
    """)

    if df is not None:
        st.success("Dataset loaded successfully")
        st.write("Shape:", df.shape)
    else:
        st.warning("Dataset not found (dashboard will still work)")

# ================= DATA ANALYTICS =================
elif menu == "Data Analytics":
    st.subheader("📈 Data Analytics")

    if df is not None:
        st.write("Preview of dataset:")
        st.dataframe(df.head())

        # Try basic visualization safely
        if "sales" in df.columns:
            st.line_chart(df["sales"])
        else:
            st.info("No 'sales' column found for visualization")

        st.write("Basic statistics:")
        st.write(df.describe())

    else:
        st.error("Dataset not available. Please add data/train.csv")

# ================= PREDICTION =================
elif menu == "Prediction":
    st.subheader("🤖 Sales Prediction")

    if model is None:
        st.error("Model not loaded")
    else:
        store = st.number_input("Store ID", min_value=1)
        item = st.number_input("Item ID", min_value=1)
        day = st.number_input("Day", 1, 31)
        month = st.number_input("Month", 1, 12)
        year = st.number_input("Year", 2000, 2100)

        if st.button("Predict Sales"):
            try:
                input_data = np.array([[store, item, day, month, year]])
                prediction = model.predict(input_data)

                st.success(f"📦 Predicted Sales: {prediction[0]}")

            except Exception as e:
                st.error("Prediction failed. Feature mismatch issue likely.")
                st.code(str(e))

# ================= ABOUT =================
elif menu == "About":
    st.subheader("ℹ About This Project")

    st.write("""
    This project is a Machine Learning-based Sales Forecasting Dashboard built using:

    - Streamlit
    - XGBoost
    - Pandas
    - NumPy

    Created for academic/project demonstration.
    """)
