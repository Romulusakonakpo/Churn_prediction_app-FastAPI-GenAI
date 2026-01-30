import streamlit as st
import requests
import base64

# --- API configuration ---
API_URL = "http://localhost:8000/predict"

# --- Page config ---
st.set_page_config(
    page_title="Churn Prediction",
    layout="wide"
)

# --- Global style ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    h1, h2, h3 {
        font-weight: bold;
    }
    .metric-value {
        font-size: 28px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- UI STYLE (Background) ----------
def add_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(0,0,0,0.2);
            background-blend-mode: multiply;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Optionnel : image légère de fond ---
add_background("assets/churn1.jpg")

# ---------- HEADER WITH LOGO ----------
col1, col2 = st.columns([1, 6])
with col1:
    st.image("assets/logo_noir.png", width=90)  # logo entreprise
with col2:
    st.markdown("<h1>Churn Prediction Tool</h1>", unsafe_allow_html=True)
    st.write(
        "Enter this client’s information to estimate churn risk and receive a personalized explanation."
    )

# ---------- INPUTS ----------
st.markdown("<h2>Client Information</h2>", unsafe_allow_html=True)

tenure = st.number_input(
    "Month's number (months)",
    min_value=0,
    max_value=120,
    value=12,
    help="Number of months this client has been with the company."
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=500.0,
    value=70.0,
    help="Amount billed every month for this client’s subscription."
)

# --- Logical minimum for TotalCharges ---
min_total_charges = tenure * monthly_charges
total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=50000.0,
    value=max(min_total_charges, 800.0),
    help="Total amount paid by this client since the beginning of the contract."
)

if total_charges < min_total_charges:
    st.warning(
        f"Total Charges cannot be lower than Tenure × Monthly Charges "
        f"({min_total_charges:.2f}). Adjusting automatically."
    )
    total_charges = min_total_charges

internet_service = st.selectbox(
    "Internet Service",
    options=["DSL", "Fiber optic", "No"],
    help="Type of internet service subscribed by this client."
)

contract = st.selectbox(
    "Contract Type",
    options=["Month-to-month", "One year", "Two year"],
    help="Contract duration currently chosen by this client."
)

# ---------- SUBMIT ----------
st.divider()

if st.button("🔍 Predict churn risk"):
    st.info(
        "Generating the explanation may take ~1 minute due to the lightweight LLM model used."
    )

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

            st.markdown("<h2>📈 Prediction Result</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='metric-value'>{result['probability']:.2%}</span>",
                unsafe_allow_html=True
            )

            st.markdown("<h2>🧠 Interpretation</h2>", unsafe_allow_html=True)
            st.write(result["explanation"])

        else:
            st.error(
                f"API error {response.status_code}: {response.text}"
            )

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
