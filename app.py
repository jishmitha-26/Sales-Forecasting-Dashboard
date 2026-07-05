import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "xgboost_sales_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# ================= LOAD MODEL =================
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

model = load_model()

# ================= LOAD DATA =================
def load_data():
    try:
        if os.path.exists(DATA_PATH):
            return pd.read_csv(DATA_PATH)
        return None
    except:
        return None

df = load_data()

# ================= APP UI =================
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Sales Forecasting Dashboard")

# ================= SIDEBAR NAV =================
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Analytics", "Prediction", "About"]
)

# ================= OVERVIEW =================
def show_overview():
    st.subheader("📌 Overview")

    st.write("""
    Welcome to the Sales Forecasting Dashboard.

    This project uses a trained Machine Learning model (XGBoost)
    to predict future sales based on input features.
    """)

    col1, col2 = st.columns(2)

    with col1:
        if df is not None:
            st.success("Dataset Loaded")
            st.write("Shape:", df.shape)
        else:
            st.warning("Dataset not available")

    with col2:
        if model is not None:
            st.success("Model Loaded")
        else:
            st.error("Model not loaded")

# ================= DATA ANALYTICS =================
def show_analytics():
    st.subheader("📈 Data Analytics")

    if df is None:
        st.error("Dataset not found in data/train.csv")
        return

    st.write("Preview:")
    st.dataframe(df.head())

    st.write("Statistics:")
    st.write(df.describe())

    # safe visualization
    if "sales" in df.columns:
        st.line_chart(df["sales"])
    else:
        st.info("No 'sales' column found for plotting")

# ================= PREDICTION =================
def show_prediction():
    st.subheader("🤖 Sales Prediction")

    if model is None:
        st.error("Model not loaded")
        return

    col1, col2 = st.columns(2)

    with col1:
        store = st.number_input("Store ID", 1)
        item = st.number_input("Item ID", 1)
        day = st.number_input("Day", 1, 31)

    with col2:
        month = st.number_input("Month", 1, 12)
        year = st.number_input("Year", 2000, 2100)

    if st.button("Predict Sales"):
        try:
            input_data = np.array([[store, item, day, month, year]])
            prediction = model.predict(input_data)

            st.success(f"📦 Predicted Sales: {prediction[0]}")

        except Exception as e:
            st.error("Prediction failed (feature mismatch likely)")
            st.code(str(e))

# ================= ABOUT =================
def show_about():
    st.subheader("ℹ About Project")

    st.write("""
    Sales Forecasting Dashboard built using:

    - Streamlit  
    - XGBoost Model  
    - Pandas  
    - NumPy  

    Features:
    - Data Analysis  
    - Visualization  
    - Machine Learning Prediction  
    """)

# ================= ROUTING =================
if menu == "Overview":
    show_overview()

elif menu == "Data Analytics":
    show_analytics()

elif menu == "Prediction":
    show_prediction()

elif menu == "About":
    show_about()
