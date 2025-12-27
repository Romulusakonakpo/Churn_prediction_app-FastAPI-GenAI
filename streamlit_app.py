import streamlit as st
import requests

# --- API configuration ---
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Churn Prediction Tool")
st.write("Enter this client’s information to estimate churn risk and receive a personalized explanation.")

# --- Inputs aligned with the 5 features ---
tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=500.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=20000.0,
    value=800.0
)

internet_service = st.selectbox(
    "Internet Service",
    options=["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Contract Type",
    options=["Month-to-month", "One year", "Two year"]
)

# --- Submit ---
if st.button("Predict churn risk"):
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService": internet_service,
        "Contract": contract
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=180)

        if response.status_code == 200:
            result = response.json()

            st.subheader("📈 Prediction Result")

            st.metric(
                label="Churn Probability",
                value=f"{result['probability']:.2%}"
            )

            st.subheader("🧠 Interpretation")
            st.write(result["explanation"])

        else:
            st.error(f"API error {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
