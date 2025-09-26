# streamlit_app.py
# Full Streamlit app for GBSA survival model prediction
# - Loads model_features.json for the model's expected feature order
# - Loads trained GBSA model from models/final_gbsa_model.pkl
# - Builds a UI for user to input customer attributes (contract, internet, payment, binary flags,
#   all 6 addon yes/no features to compute NumSecurityServices and NumStreamingServices,
#   and the 5 synthetic features)
# - Constructs a single-row DataFrame matching model_features and predicts:
#     * risk score (higher = higher risk)
#     * survival curve (plotted)
#     * median predicted survival time
#
# Requirements:
# pip install streamlit scikit-survival joblib pandas numpy matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sksurv.util import Surv

# ---------------------------
# CONFIG: file paths (adjust if needed)
# ---------------------------
MODEL_PATH = "models/final_gbsa_model.pkl"
FEATURES_JSON = "models/model_features.json"

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    model = joblib.load(path)
    return model

@st.cache_data(show_spinner=False)
def load_model_features(path):
    with open(path, "r") as f:
        features = json.load(f)
    # expect features to be list of strings
    return features

def build_input_row(features_list, inputs):
    """
    features_list: ordered list of model features (strings)
    inputs: dict mapping logical variable names to values (not necessarily feature names)
    Returns: pd.DataFrame single row with columns = features_list (in same order)
    """
    # initialize zeros
    row = pd.Series(0, index=features_list, dtype=float)

    # Binary features that are part of feature list (expected named exactly)
    binary_map = {
        "SeniorCitizen": inputs.get("SeniorCitizen", 0),
        "Partner": inputs.get("Partner", 0),
        "Dependents": inputs.get("Dependents", 0),
        "PaperlessBilling": inputs.get("PaperlessBilling", 0),
        "discount_received": inputs.get("discount_received", 0),
        "no_complaint_flag": inputs.get("no_complaint_flag", 0),
    }
    for k,v in binary_map.items():
        if k in row.index:
            row[k] = float(v)

    # Numeric features
    numeric_map = {
        "duration": inputs.get("duration", 0.0),  # usually 1..72 (months)
        "num_complaints": inputs.get("num_complaints", 0),
        "last_complaint_days_ago": inputs.get("last_complaint_days_ago", 0.0),
        "data_usage_gb": inputs.get("data_usage_gb", 0.0),
        "late_payments_count": inputs.get("late_payments_count", 0),
        "NumSecurityServices": inputs.get("NumSecurityServices", 0),
        "NumStreamingServices": inputs.get("NumStreamingServices", 0),
        # add MonthlyCharges or TotalCharges if model expects them (check features_list)
    }
    for k,v in numeric_map.items():
        if k in row.index:
            row[k] = float(v)

    # One-hot categorical features: we expect features list contains dummy column names like:
    # 'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
    # 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    # The reference categories are dropped (e.g., DSL, Month-to-month, Bank transfer)
    # So set the appropriate dummy column to 1 if present.
    # inputs provides categorical choices as strings.
    cat_map = {
        "InternetService": inputs.get("InternetService", None),
        "Contract": inputs.get("Contract", None),
        "PaymentMethod": inputs.get("PaymentMethod", None)
    }

    # Contract
    contract_val = cat_map["Contract"]
    if contract_val is not None:
        # expected dummy names
        if contract_val == "One year":
            name = "Contract_One year"
            if name in row.index: row[name] = 1.0
        elif contract_val == "Two year":
            name = "Contract_Two year"
            if name in row.index: row[name] = 1.0
        # Month-to-month is reference -> nothing to set

    # InternetService
    ist = cat_map["InternetService"]
    if ist is not None:
        if ist == "Fiber optic":
            name = "InternetService_Fiber optic"
            if name in row.index: row[name] = 1.0
        elif ist == "No":
            name = "InternetService_No"
            if name in row.index: row[name] = 1.0
        # DSL was reference -> nothing to set

    # PaymentMethod
    pm = cat_map["PaymentMethod"]
    if pm is not None:
        if pm == "Credit card (automatic)":
            name = "PaymentMethod_Credit card (automatic)"
            if name in row.index: row[name] = 1.0
        elif pm == "Electronic check":
            name = "PaymentMethod_Electronic check"
            if name in row.index: row[name] = 1.0
        elif pm == "Mailed check":
            name = "PaymentMethod_Mailed check"
            if name in row.index: row[name] = 1.0
        # Bank transfer (automatic) was reference -> nothing to set

    # Return as single-row DataFrame in the same column order
    return pd.DataFrame([row.values], columns=row.index)

def plot_survival_functions(surv_funcs, title="Predicted Survival Curve"):
    plt.figure(figsize=(8,5))
    for fn in surv_funcs:
        # fn is a StepFunction-like object with .x and .y attributes (sk-surv)
        x = fn.x
        y = fn.y
        plt.step(x, y, where="post", alpha=0.8)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Months since start")
    plt.ylabel("Survival probability")
    plt.title(title)
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

# ---------------------------
# Load model & feature list
# ---------------------------
st.set_page_config(page_title="Churn Survival Predictor (GBSA)", layout="wide")
st.title("Customer Survival & Churn Risk — GBSA (Production)")

# Load artifacts
try:
    gbsa_model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load GBSA model from {MODEL_PATH}: {e}")
    st.stop()

try:
    model_features = load_model_features(FEATURES_JSON)
except Exception as e:
    st.error(f"Could not load feature list from {FEATURES_JSON}: {e}")
    st.stop()

st.markdown("**Model loaded. Fill the customer fields in the sidebar and click Predict.**")

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Customer Input")

# Basic customer details / durations
duration = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=240, value=12, step=1)
# event not needed (we predict), but we require duration as a covariate

# Contract & service & payment (original categorical)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

# Binary flags
senior = st.sidebar.checkbox("Senior Citizen", value=False)
partner = st.sidebar.checkbox("Partner", value=False)
dependents = st.sidebar.checkbox("Dependents", value=False)
paperless = st.sidebar.checkbox("Paperless Billing", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Add-on Services (Yes / No)")
st.sidebar.markdown("Answer Yes/No for the following 6 add-ons. They will be summed into NumSecurityServices and NumStreamingServices.")

# Define addon inputs: 4 security services + 2 streaming
sec_online_security = st.sidebar.radio("Online Security?", ("No", "Yes"))
sec_online_backup = st.sidebar.radio("Online Backup?", ("No", "Yes"))
sec_device_protection = st.sidebar.radio("Device Protection?", ("No", "Yes"))
sec_tech_support = st.sidebar.radio("Tech Support?", ("No", "Yes"))

stream_tv = st.sidebar.radio("Streaming TV?", ("No", "Yes"))
stream_movies = st.sidebar.radio("Streaming Movies?", ("No", "Yes"))

# Compute NumSecurityServices and NumStreamingServices
num_security = sum([1 if v == "Yes" else 0 for v in [sec_online_security, sec_online_backup, sec_device_protection, sec_tech_support]])
num_streaming = sum([1 if v == "Yes" else 0 for v in [stream_tv, stream_movies]])

# Synthetic features
st.sidebar.markdown("---")
st.sidebar.subheader("Behavioral Inputs (Synthetic features)")

num_complaints = st.sidebar.number_input("Number of complaints", min_value=0, max_value=50, value=0, step=1)
# if num_complaints==0, last_complaint_days_ago will be auto 0 and flag set
if num_complaints == 0:
    last_complaint_days_ago = 0.0
else:
    last_complaint_days_ago = st.sidebar.number_input("Days since last complaint", min_value=0.0, max_value=5000.0, value=90.0, step=1.0)

data_usage_gb = st.sidebar.number_input("Data usage (GB/month)", min_value=0.0, max_value=10000.0, value=20.0, step=0.1)
late_payments_count = st.sidebar.number_input("Late payments count", min_value=0, max_value=100, value=0, step=1)
discount_received = st.sidebar.checkbox("Discount received previously", value=False)

# auto-compute no_complaint_flag
no_complaint_flag = 1 if num_complaints == 0 else 0

# Sidebar CTA
st.sidebar.markdown("---")
if st.sidebar.button("Predict survival & risk"):
    # Build inputs dict
    inputs = {
        "duration": float(duration),
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "SeniorCitizen": 1 if senior else 0,
        "Partner": 1 if partner else 0,
        "Dependents": 1 if dependents else 0,
        "PaperlessBilling": 1 if paperless else 0,
        "NumSecurityServices": int(num_security),
        "NumStreamingServices": int(num_streaming),
        "num_complaints": int(num_complaints),
        "last_complaint_days_ago": float(last_complaint_days_ago),
        "data_usage_gb": float(data_usage_gb),
        "late_payments_count": int(late_payments_count),
        "discount_received": 1 if discount_received else 0,
        "no_complaint_flag": int(no_complaint_flag)
    }

    # Build feature-row DataFrame in correct order
    X_row = build_input_row(model_features, inputs)

    st.write("### Input features (model order)")
    st.dataframe(X_row.T.rename(columns={0: "value"}))

    # Predict risk score and survival function
    try:
        # risk score: higher means higher predicted hazard (GBSA.predict returns risk score)
        risk_score = gbsa_model.predict(X_row)[0]
        st.write(f"**Risk score (higher = higher risk):** {risk_score:.4f}")

        # survival function
        surv_fns = gbsa_model.predict_survival_function(X_row)
        # Plot survival function
        st.write("#### Predicted survival curve")
        plot_survival_functions(surv_fns, title="Predicted Survival Curve (GBSA)")

        # approximate median survival: find time where survival <= 0.5
        fn = surv_fns[0]
        # fn.x (times), fn.y (survival probs)
        times = fn.x
        surv_probs = fn.y
        # find first time where survival <= 0.5
        if np.any(surv_probs <= 0.5):
            median_idx = np.argmax(surv_probs <= 0.5)
            median_time = times[median_idx]
            st.write(f"**Predicted median survival (months):** {median_time:.1f}")
        else:
            st.write("**Predicted median survival (months):** > max modelled time")

        # Provide short business suggestion based on simple heuristics
        st.markdown("### Quick business suggestion")
        suggestions = []
        if num_complaints >= 1 and last_complaint_days_ago <= 90:
            suggestions.append("- Customer had a recent complaint → prioritize retention call + service recovery.")
        if data_usage_gb >= 40:
            suggestions.append("- High data usage → offer premium service check or prioritized tech support.")
        if late_payments_count >= 1:
            suggestions.append("- Late payments detected → offer flexible billing or payment reminders.")
        if discount_received:
            suggestions.append("- Customer already received a discount; combine with service improvements rather than another pure discount.")
        if inputs["Contract"] == "Month-to-month":
            suggestions.append("- Consider targeted offer to convert to 1-year contract (bundle + discount).")
        if len(suggestions) == 0:
            suggestions.append("- No immediate red flags; consider standard retention monitoring.")

        for s in suggestions:
            st.write(s)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------
# Footer: notes & instructions
# ---------------------------
st.markdown("---")
st.markdown(
    """
    **Notes:**  
    - This app uses the production GBSA model saved at `models/final_gbsa_model.pkl`.  
    - The app expects `model_features.json` to contain the exact feature column names (in order) used during training.  
    - The input UI converts human-friendly inputs into the model's one-hot / numeric features (reference categories are left as zeros).  
    - For batch scoring, prepare a CSV with the same columns as `model_features.json` and load / predict externally.
    """
)
