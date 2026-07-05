import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# FILE PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "..", "xgboost_sales_model.pkl")
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "train.csv")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

train = pd.read_csv(TRAIN_PATH)

train["date"] = pd.to_datetime(train["date"])

# ---------------------------------------------------
# CREATE FEATURES
# ---------------------------------------------------

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["weekday"] = train["date"].dt.weekday

train["lag_1"] = train.groupby(["store_nbr", "family"])["sales"].shift(1)

train["rolling_7"] = (
    train.groupby(["store_nbr", "family"])["sales"]
    .rolling(7)
    .mean()
    .reset_index(level=[0,1], drop=True)
)

train["lag_1"] = train["lag_1"].fillna(0)
train["rolling_7"] = train["rolling_7"].fillna(0)

# ---------------------------------------------------
# PRODUCT FAMILY MAPPING
# ---------------------------------------------------

family_mapping = {
    'AUTOMOTIVE':0,
    'BABY CARE':1,
    'BEAUTY':2,
    'BEVERAGES':3,
    'BOOKS':4,
    'BREAD/BAKERY':5,
    'CELEBRATION':6,
    'CLEANING':7,
    'DAIRY':8,
    'DELI':9,
    'EGGS':10,
    'FROZEN FOODS':11,
    'GROCERY I':12,
    'GROCERY II':13,
    'HARDWARE':14,
    'HOME AND KITCHEN I':15,
    'HOME AND KITCHEN II':16,
    'HOME APPLIANCES':17,
    'HOME CARE':18,
    'LADIESWEAR':19,
    'LAWN AND GARDEN':20,
    'LINGERIE':21,
    'LIQUOR,WINE,BEER':22,
    'MAGAZINES':23,
    'MEATS':24,
    'PERSONAL CARE':25,
    'PET SUPPLIES':26,
    'PLAYERS AND ELECTRONICS':27,
    'POULTRY':28,
    'PREPARED FOODS':29,
    'PRODUCE':30,
    'SCHOOL AND OFFICE SUPPLIES':31,
    'SEAFOOD':32
}

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📊 Sales Forecasting Dashboard")

st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("""
### Model

- XGBoost Regressor

### Performance

- R² Score : **95.29%**
- RMSE : **237.70**

### Dataset

Store Sales Time Series Forecasting
""")

st.sidebar.divider()

st.sidebar.info(
    "Enter Store Number, Product Family and Date to predict future sales."
)

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("📈 Store Sales Forecasting Dashboard")

st.write(
    "Predict future sales using an XGBoost Machine Learning model."
)
# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

st.header("🔮 Sales Prediction")

col1, col2 = st.columns(2)

with col1:
    store = st.number_input(
        "🏪 Store Number",
        min_value=1,
        max_value=54,
        value=1
    )

    family_name = st.selectbox(
        "📦 Product Family",
        list(family_mapping.keys())
    )

    family = family_mapping[family_name]

with col2:
    selected_date = st.date_input("📅 Select Date")

year = selected_date.year
month = selected_date.month
day = selected_date.day
weekday = selected_date.weekday()

# Use average lag values
lag_1 = train["lag_1"].mean()
rolling_7 = train["rolling_7"].mean()

if st.button("🚀 Predict Sales", use_container_width=True):

    input_data = np.array([[
        store,
        family,
        year,
        month,
        day,
        weekday,
        lag_1,
        rolling_7
    ]])

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    st.success(f"### 📈 Predicted Sales: {prediction:.2f}")

    st.subheader("Prediction Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("🏪 Store", store)
        st.metric("📦 Product", family_name)

    with c2:
        st.metric("📅 Date", str(selected_date))
        st.metric("💰 Predicted Sales", f"{prediction:.2f}")

st.divider()

# ---------------------------------------------------
# PROJECT METRICS
# ---------------------------------------------------

st.header("📊 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Stores", train["store_nbr"].nunique())

with col2:
    st.metric("Product Families", len(family_mapping))

with col3:
    st.metric("Model R²", "95.29%")

st.divider()
# ---------------------------------------------------
# DAILY SALES TREND
# ---------------------------------------------------

st.header("📈 Daily Sales Trend")

daily_sales = (
    train.groupby("date")["sales"]
    .sum()
    .reset_index()
)

st.line_chart(
    daily_sales.set_index("date")
)

st.divider()

# ---------------------------------------------------
# TOP 10 PRODUCT FAMILIES
# ---------------------------------------------------

st.header("🏆 Top 10 Product Families by Sales")

family_sales = (
    train.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(family_sales)

st.divider()

# ---------------------------------------------------
# TOP 10 STORES
# ---------------------------------------------------

st.header("🏪 Top 10 Stores by Total Sales")

store_sales = (
    train.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(store_sales)

st.divider()

# ---------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------



# ---------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
"""
### About this Project

This Sales Forecasting Dashboard was built using:

-  Python
-  Pandas & NumPy
-  XGBoost Machine Learning
-  Streamlit

Model Performance

-  R² Score : **95.29%**
-  RMSE : **237.70**

Developed as an end-to-end Machine Learning project.
"""
)
