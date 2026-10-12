import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_model.pkl")


model = load_model()


# --------------------------------------------------
# Risk Level Function
# --------------------------------------------------

def get_risk_level(probability):

    if probability < 0.30:
        return "Low"

    elif probability < 0.60:
        return "Medium"

    else:
        return "High"


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("📊 AI-Powered Customer Churn Prediction")

st.write(
    "Enter customer information below to predict "
    "the probability of customer churn and identify "
    "the customer's risk level."
)


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


with col2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
):

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]

    probability_percentage = probability * 100

    risk = get_risk_level(probability)


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.header("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        if prediction == 1:
            st.error("⚠️ Customer Will Churn")
        else:
            st.success("✅ Customer Will Not Churn")


    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability_percentage:.2f}%"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            risk
        )


    # --------------------------------------------------
    # Risk Explanation
    # --------------------------------------------------

    st.subheader("Risk Analysis")

    if risk == "High":

        st.warning(
            "This customer has a high predicted churn risk. "
            "Customer retention actions may be considered."
        )

    elif risk == "Medium":

        st.info(
            "This customer has a medium predicted churn risk. "
            "Monitoring and targeted engagement may be considered."
        )

    else:

        st.success(
            "This customer has a low predicted churn risk."
        )


    # --------------------------------------------------
    # Customer Data
    # --------------------------------------------------

    with st.expander("View Customer Input Data"):

        st.dataframe(
            customer_data,
            use_container_width=True
        )