# app.py — Production-ready Streamlit app for GBSA churn-survival model
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
import shap 

# Config / paths
MODEL_PATH = Path("models/final_gbsa_model.pkl")
FEATURES_JSON = Path("models/model_features.json")
PLOTS_DIR = Path("plots")
PERMUTATION_PLOT = PLOTS_DIR / "gbsa_feature_importance.png"
SAMPLE_SURVIVAL_PLOT = PLOTS_DIR / "gbsa_sample_predicted_survival_curves.png"
GROUP_SURVIVAL_PLOT = PLOTS_DIR / "gbsa_group_survival_curves.png"

st.set_page_config(page_title="Churn Survival — GBSA", layout="wide")
st.title("Customer Lifetime & Churn Risk — Production GBSA")

# Utilities
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_features(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def build_input_row(features_list, inputs):
    row = pd.Series(0.0, index=features_list, dtype=float)

    # binary flags
    for k in ["SeniorCitizen","Partner","Dependents","PaperlessBilling","discount_received","no_complaint_flag"]:
        if k in row.index and k in inputs:
            row[k] = float(inputs[k])

    # numeric features
    for k in ["duration","num_complaints","last_complaint_days_ago","data_usage_gb","late_payments_count","NumSecurityServices","NumStreamingServices"]:
        if k in row.index and k in inputs:
            row[k] = float(inputs[k])

    # categorical one-hots
    contract = inputs.get("Contract")
    if contract == "One year" and "Contract_One year" in row.index:
        row["Contract_One year"] = 1.0
    elif contract == "Two year" and "Contract_Two year" in row.index:
        row["Contract_Two year"] = 1.0

    ist = inputs.get("InternetService")
    if ist == "Fiber optic" and "InternetService_Fiber optic" in row.index:
        row["InternetService_Fiber optic"] = 1.0
    elif ist == "No" and "InternetService_No" in row.index:
        row["InternetService_No"] = 1.0

    pm = inputs.get("PaymentMethod")
    if pm == "Credit card (automatic)" and "PaymentMethod_Credit card (automatic)" in row.index:
        row["PaymentMethod_Credit card (automatic)"] = 1.0
    elif pm == "Electronic check" and "PaymentMethod_Electronic check" in row.index:
        row["PaymentMethod_Electronic check"] = 1.0
    elif pm == "Mailed check" and "PaymentMethod_Mailed check" in row.index:
        row["PaymentMethod_Mailed check"] = 1.0

    return pd.DataFrame([row.values], columns=row.index)

def plot_survival(surv_funcs, title="Predicted Survival Curve"):
    fig, ax = plt.subplots(figsize=(8,4.5))
    for fn in surv_funcs:
        ax.step(fn.x, fn.y, where="post", alpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Months since start")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)
    plt.close(fig)

# Load artifacts
try:
    gbsa = load_model(MODEL_PATH)
    model_features = load_features(FEATURES_JSON)
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

# Input form
st.header("Enter Customer Profile")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Basic & Contract Info")
    duration = st.number_input("Tenure (months)", 0, 240, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
    senior = st.checkbox("Senior Citizen", False)
    partner = st.checkbox("Partner", False)
    dependents = st.checkbox("Dependents", False)
    paperless = st.checkbox("Paperless Billing", True)

with col2:
    st.subheader("Add-ons (summed)")
    sec_online_security = st.selectbox("Online Security", ["No","Yes"])
    sec_online_backup = st.selectbox("Online Backup", ["No","Yes"])
    sec_device_protection = st.selectbox("Device Protection", ["No","Yes"])
    sec_tech_support = st.selectbox("Tech Support", ["No","Yes"])
    stream_tv = st.selectbox("Streaming TV", ["No","Yes"])
    stream_movies = st.selectbox("Streaming Movies", ["No","Yes"])

    st.subheader("Behavioral Inputs")
    num_complaints = st.number_input("Number of complaints", 0, 100, 0)
    if num_complaints == 0:
        last_complaint_days_ago = 0.0
        no_complaint_flag = 1
    else:
        last_complaint_days_ago = st.number_input("Days since last complaint", 0, 5000, 90)
        no_complaint_flag = 0

    data_usage_gb = st.number_input("Data usage (GB/month)", 0.0, 10000.0, 20.0, step=0.1)
    late_payments_count = st.number_input("Late payments count", 0, 100, 0)
    discount_received = st.checkbox("Discount received previously", False)

num_security = sum([1 for v in (sec_online_security, sec_online_backup, sec_device_protection, sec_tech_support) if v == "Yes"])
num_streaming = sum([1 for v in (stream_tv, stream_movies) if v == "Yes"])

# Prediction
if st.button("Predict survival & risk", type="primary"):
    inputs = {
        "duration": float(duration),
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "SeniorCitizen": 1 if senior else 0,
        "Partner": 1 if partner else 0,
        "Dependents": 1 if dependents else 0,
        "PaperlessBilling": 1 if paperless else 0,
        "NumSecurityServices": num_security,
        "NumStreamingServices": num_streaming,
        "num_complaints": num_complaints,
        "last_complaint_days_ago": last_complaint_days_ago,
        "data_usage_gb": data_usage_gb,
        "late_payments_count": late_payments_count,
        "discount_received": 1 if discount_received else 0,
        "no_complaint_flag": no_complaint_flag
    }

    try:
        X_row = build_input_row(model_features, inputs)

        risk_score = gbsa.predict(X_row)[0]
        surv_fns = gbsa.predict_survival_function(X_row)

        fn = surv_fns[0]
        median_time = None
        if np.any(fn.y <= 0.5):
            median_time = fn.x[np.argmax(fn.y <= 0.5)]

        st.success("Prediction complete")
        st.subheader("Model Output")
        st.metric("Relative churn risk score", f"{risk_score:.3f}", help="Higher score = higher likelihood of earlier churn compared to peers")
        if median_time:
            st.metric("Predicted median survival (months)", f"{median_time:.1f}")
        else:
            st.metric("Predicted median survival (months)", "Likely > maximum observed tenure")

        st.markdown("#### Predicted survival curve")
        plot_survival(surv_fns)

        # Business suggestions
        st.markdown("#### Suggested Actions")
        tips = []
        if num_complaints and last_complaint_days_ago <= 90:
            tips.append("Recent complaint: immediate service recovery needed.")
        if data_usage_gb >= 40:
            tips.append("High usage: offer premium support or better plan.")
        if late_payments_count:
            tips.append("Late payments: provide billing flexibility.")
        if discount_received:
            tips.append("Already discounted: combine support/loyalty benefits, not just price cuts.")
        if contract == "Month-to-month":
            tips.append("Month-to-month: propose 1-year bundle with perks.")
        if not tips:
            tips.append("No immediate red flags: standard retention cadence.")
        for t in tips:
            st.write("- " + t)

        # SHAP explanation
        st.markdown("#### Why this prediction?")
        try:
            explainer = shap.KernelExplainer(gbsa.predict, pd.DataFrame(np.zeros((1, X_row.shape[1])), columns=X_row.columns))
            shap_values = explainer.shap_values(X_row, nsamples=100)
            exp = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_row.iloc[0].values,
                feature_names=X_row.columns
            )
            
            fig, ax = plt.subplots(figsize=(8,6))
            shap.waterfall_plot(exp, max_display=10, show=False)
            st.pyplot(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
        except Exception as se:
            st.info(f"SHAP explanation not available: {se}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Static visuals
st.markdown("---")
st.header("Model Diagnostics (precomputed)")

cols = st.columns(3)
with cols[0]:
    st.subheader("Permutation Importance")
    if PERMUTATION_PLOT.exists():
        st.image(str(PERMUTATION_PLOT), use_container_width=True)

with cols[1]:
    st.subheader("Sample Survival Curves")
    if SAMPLE_SURVIVAL_PLOT.exists():
        st.image(str(SAMPLE_SURVIVAL_PLOT), use_container_width=True)

with cols[2]:
    st.subheader("High vs Low Risk Groups")
    if GROUP_SURVIVAL_PLOT.exists():
        st.image(str(GROUP_SURVIVAL_PLOT), use_container_width=True)
