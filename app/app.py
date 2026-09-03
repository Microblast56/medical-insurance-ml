import streamlit as st
import joblib
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import preprocess_input


# Load trained model
model_path = PROJECT_ROOT / "models" / "insurance_model.pkl"
model = joblib.load(model_path)


# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)


# Title
st.title("Medical Insurance Cost Predictor")

st.write(
    "A basic machine learning application that estimates "
    "medical insurance charges from demographic and "
    "insurance-related information."
)

st.divider()


# Personal information
st.subheader("Personal Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)


# Insurance information
st.subheader("Insurance Information")

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)


st.divider()


# Prediction
if st.button("Predict Insurance Charges", type="primary"):

    input_data = preprocess_input(
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Medical Insurance Charges: "
        f"{prediction:,.2f}"
    )

    st.caption(
        "This is an estimated machine learning prediction "
        "and not an actual insurance quote."
    )