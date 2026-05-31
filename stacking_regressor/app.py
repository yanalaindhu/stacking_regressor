import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="🛡️",
    layout="centered"
)

st.title("Insurance Premium Prediction Using Stacking Regressor")

# -----------------------------
# Get Current Directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stacking_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "encoders.pkl"
)

# -----------------------------
# Debug Information
# -----------------------------
st.write("Current Directory:", BASE_DIR)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.stop()

if not os.path.exists(ENCODER_PATH):
    st.error(f"Encoder file not found:\n{ENCODER_PATH}")
    st.stop()

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open(MODEL_PATH, "rb"))
encoders = pickle.load(open(ENCODER_PATH, "rb"))

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Premium"):

    sex_encoded = encoders["sex"].transform([sex])[0]
    smoker_encoded = encoders["smoker"].transform([smoker])[0]
    region_encoded = encoders["region"].transform([region])[0]

    data = [[
        age,
        sex_encoded,
        bmi,
        children,
        smoker_encoded,
        region_encoded
    ]]

    prediction = model.predict(data)

    st.success(
        f"Estimated Insurance Charges: ₹ {prediction[0]:,.2f}"
    )
