import streamlit as st
import pickle

st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="🛡️"
)

st.title("Insurance Premium Prediction Using Stacking Regressor")

# Load Model
model = pickle.load(
    open("models/stacking_model.pkl", "rb")
)

encoders = pickle.load(
    open("models/encoders.pkl", "rb")
)

# Inputs
age = st.number_input("Age", 18, 100)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

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