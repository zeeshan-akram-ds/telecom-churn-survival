# app.py — Production-ready Streamlit app for GBSA churn-survival model
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Optional imports for SHAP (only used if available)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# config / paths (adjust if needed)
MODEL_PATH = Path("models/final_gbsa_model.pkl")
FEATURES_JSON = Path("models/model_features.json")
PLOTS_DIR = Path("plots")
PERMUTATION_PLOT = PLOTS_DIR / "permutation_importance.png"
SAMPLE_SURVIVAL_PLOT = PLOTS_DIR / "sample_survival_curves.png"
GROUP_SURVIVAL_PLOT = PLOTS_DIR / "group_survival_curves.png"

st.set_page_config(page_title="Churn Survival — GBSA", layout="wide")
st.title("Customer Lifetime & Churn Risk — Production GBSA")

# -------------------------
# Utilities & loaders
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_features(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Feature JSON not found at {path}")
    with open(path, "r") as f:
        feats = json.load(f)
    return feats

def build_input_row(features_list, inputs):
    """Return single-row DataFrame aligned to features_list (order preserved)."""
    row = pd.Series(0.0, index=features_list, dtype=float)

    # binary flags
    for k in ["SeniorCitizen","Partner","Dependents","PaperlessBilling","discount_received","no_complaint_flag"]:
        if k in row.index and k in inputs:
            row[k] = float(inputs[k])

    # numeric features
    for k in ["duration","num_complaints","last_complaint_days_ago","data_usage_gb","late_payments_count","NumSecurityServices","NumStreamingServices"]:
        if k in row.index and k in inputs:
            row[k] = float(inputs[k])

    # categorical one-hot handling (Contract, InternetService, PaymentMethod)
    # Contract: Month-to-month (ref), One year, Two year
    contract = inputs.get("Contract")
    if contract == "One year" and "Contract_One year" in row.index:
        row["Contract_One year"] = 1.0
    elif contract == "Two year" and "Contract_Two year" in row.index:
        row["Contract_Two year"] = 1.0

    # InternetService: DSL (ref), Fiber optic, No
    ist = inputs.get("InternetService")
    if ist == "Fiber optic" and "InternetService_Fiber optic" in row.index:
        row["InternetService_Fiber optic"] = 1.0
    elif ist == "No" and "InternetService_No" in row.index:
        row["InternetService_No"] = 1.0

    # PaymentMethod: Bank transfer (ref), Credit card (automatic), Electronic check, Mailed check
    pm = inputs.get("PaymentMethod")
    if pm == "Credit card (automatic)" and "PaymentMethod_Credit card (automatic)" in row.index:
        row["PaymentMethod_Credit card (automatic)"] = 1.0
    elif pm == "Electronic check" and "PaymentMethod_Electronic check" in row.index:
        row["PaymentMethod_Electronic check"] = 1.0
    elif pm == "Mailed check" and "PaymentMethod_Mailed check" in row.index:
        row["PaymentMethod_Mailed check"] = 1.0

    return pd.DataFrame([row.values], columns=row.index)

def plot_survival_functions_matplotlib(surv_funcs, title="Predicted Survival Curve"):
    fig, ax = plt.subplots(figsize=(8,4.5))
    for fn in surv_funcs:
        x = fn.x
        y = fn.y
        ax.step(x, y, where="post", alpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Months since start")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)
    plt.close(fig)

# -------------------------
# Load model and features (fail fast)
# -------------------------
try:
    gbsa = load_model(MODEL_PATH)
    model_features = load_features(FEATURES_JSON)
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

# -------------------------
# Main: Inputs on the main page (not sidebar)
# -------------------------
st.header("Enter Customer Profile")
st.markdown("Fill the customer's attributes below and click **Predict**. I will show the predicted survival curve, risk score and actionable suggestions.")

# Layout: two columns for inputs and a right column for quick actions/plots
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Basic & Contract Info")
    duration = st.number_input("Tenure (months)", min_value=0, max_value=240, value=12, step=1)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
    senior = st.checkbox("Senior Citizen", value=False)
    partner = st.checkbox("Partner", value=False)
    dependents = st.checkbox("Dependents", value=False)
    paperless = st.checkbox("Paperless Billing", value=True)

with col2:
    st.subheader("Add-ons (will be summed)")
    st.markdown("Select Yes for each add-on the customer has.")
    sec_online_security = st.selectbox("Online Security", ["No","Yes"], index=0)
    sec_online_backup = st.selectbox("Online Backup", ["No","Yes"], index=0)
    sec_device_protection = st.selectbox("Device Protection", ["No","Yes"], index=0)
    sec_tech_support = st.selectbox("Tech Support", ["No","Yes"], index=0)
    stream_tv = st.selectbox("Streaming TV", ["No","Yes"], index=0)
    stream_movies = st.selectbox("Streaming Movies", ["No","Yes"], index=0)

    # synthetic features
    st.subheader("Behavioral Inputs")
    num_complaints = st.number_input("Number of complaints", min_value=0, max_value=100, value=0, step=1)
    if num_complaints == 0:
        last_complaint_days_ago = 0.0
        no_complaint_flag = 1
    else:
        last_complaint_days_ago = st.number_input("Days since last complaint", min_value=0.0, max_value=5000.0, value=90.0, step=1.0)
        no_complaint_flag = 0

    data_usage_gb = st.number_input("Data usage (GB/month)", min_value=0.0, max_value=10000.0, value=20.0, step=0.1)
    late_payments_count = st.number_input("Late payments count", min_value=0, max_value=100, value=0, step=1)
    discount_received = st.checkbox("Discount received previously", value=False)

# compute addon sums
num_security = sum([1 for v in (sec_online_security, sec_online_backup, sec_device_protection, sec_tech_support) if v == "Yes"])
num_streaming = sum([1 for v in (stream_tv, stream_movies) if v == "Yes"])

# Predict button
st.markdown("---")
predict_col, info_col = st.columns([1,1])
with predict_col:
    if st.button("Predict survival & risk", type="primary"):
        # build inputs dict
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

        # build model input and predict
        try:
            X_row = build_input_row(model_features, inputs)

            # Risk score (higher means higher risk) and survival function
            risk_score = gbsa.predict(X_row)[0]
            surv_fns = gbsa.predict_survival_function(X_row)

            # median survival (approx)
            fn = surv_fns[0]
            times = fn.x
            surv_probs = fn.y
            median_time = None
            if np.any(surv_probs <= 0.5):
                median_idx = np.argmax(surv_probs <= 0.5)
                median_time = times[median_idx]

            # Display results in a tidy panel
            st.success("Prediction complete")
            st.markdown("### Model Output")
            st.metric("Risk score (higher = higher risk)", f"{risk_score:.4f}")
            if median_time is not None:
                st.metric("Predicted median survival (months)", f"{median_time:.1f}")
            else:
                st.metric("Predicted median survival (months)", "> max modelled time")

            st.markdown("#### Predicted survival curve")
            plot_survival_functions_matplotlib(surv_fns, title="Predicted Survival Curve (GBSA)")

            # Business suggestions (simple but practical heuristics)
            st.markdown("#### Quick business suggestions")
            suggs = []
            if num_complaints >= 1 and last_complaint_days_ago <= 90:
                suggs.append("Recent complaint: prioritize this customer for service recovery and a retention offer.")
            if data_usage_gb >= 40:
                suggs.append("High usage: offer premium troubleshooting / SLA benefits to reduce friction.")
            if late_payments_count >= 1:
                suggs.append("Late payments: offer flexible billing or personalized payment reminders.")
            if discount_received:
                suggs.append("Already discounted: combine technical support or loyalty benefits, not just another discount.")
            if contract == "Month-to-month":
                suggs.append("Short contract: propose a one-year bundle with incentives to increase tenure.")
            if len(suggs) == 0:
                suggs.append("No immediate flags: monitor through standard retention cadence.")

            for s in suggs:
                st.write("- " + s)

            # Optional SHAP explainability (run only on demand)
            with info_col:
                st.markdown("#### Model explanation")
                if SHAP_AVAILABLE:
                    if st.button("Show SHAP explanation"):
                        try:
                            # Best-effort SHAP: use TreeExplainer if supported, else KernelExplainer fallback (slow)
                            explainer = None
                            try:
                                explainer = shap.TreeExplainer(gbsa)
                            except Exception:
                                explainer = shap.Explainer(gbsa.predict, X_row)  # fallback
                            shap_values = explainer(X_row)
                            # Use shap.plots.bar or summary_plot
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            fig = shap.plots.bar(shap_values, show=False)
                            st.pyplot(fig)
                            plt.clf()
                        except Exception as se:
                            st.error(f"SHAP explanation failed: {se}")
                else:
                    st.info("SHAP not installed or not available in this environment. To enable, install `shap` and restart the app.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------
# Right column: Precomputed visualizations (static images from plots/)
# -------------------------
st.markdown("---")
st.header("Model Diagnostics & Visuals (precomputed)")

img_cols = st.columns(3)
with img_cols[0]:
    st.subheader("Permutation Importance (GBSA)")
    if PERMUTATION_PLOT.exists():
        st.image(str(PERMUTATION_PLOT), use_column_width=True)
    else:
        st.info("Permutation importance image not found.")

with img_cols[1]:
    st.subheader("Sample Predicted Survival Curves")
    if SAMPLE_SURVIVAL_PLOT.exists():
        st.image(str(SAMPLE_SURVIVAL_PLOT), use_column_width=True)
    else:
        st.info("Sample survival curves image not found.")

with img_cols[2]:
    st.subheader("Group Survival Curves (High vs Low Risk)")
    if GROUP_SURVIVAL_PLOT.exists():
        st.image(str(GROUP_SURVIVAL_PLOT), use_column_width=True)
    else:
        st.info("Group survival curves image not found.")

# Footer
st.markdown("---")
st.markdown("**Notes:** This app predicts customer survival curves and risk scores using a production GBSA model. Use the Business Suggestions as immediate tactics; for large-scale scoring, run batch predictions using the saved model and the same feature engineering pipeline.")
