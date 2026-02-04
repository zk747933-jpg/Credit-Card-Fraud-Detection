import streamlit as st
import pickle
import numpy as np
from rules import rule_based_check

# Load trained ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Prediction System")
st.write("⚠️ Note: This app predicts the fraud risk of a transaction. It does not confirm actual fraud.")


st.write("Enter transaction details to check fraud risk")

# USER INPUTS (simple & realistic)
amount = st.number_input("Transaction Amount (₹)", min_value=1.0, step=100.0)
hour = st.slider("Transaction Hour (0–23)", 0, 23, 12)
location_change = st.selectbox("Transaction from new city?", ["No", "Yes"])
device_change = st.selectbox("Transaction from new device?", ["No", "Yes"])
txn_count_24h = st.slider("Transactions in last 24 hours", 1, 20, 1)

# Convert Yes/No to 0/1
location_change = 1 if location_change == "Yes" else 0
device_change = 1 if device_change == "Yes" else 0

if st.button("Check Fraud Risk"):
    # ML Prediction
    input_data = np.array([[amount, hour, location_change, device_change, txn_count_24h]])
    ml_prob = model.predict_proba(input_data)[0][1]

    # Rule-based risk
    rule_risk = rule_based_check(
        amount, hour, location_change, device_change, txn_count_24h
    )

    # Final risk (hybrid)
    final_risk = round((ml_prob * 0.6 + rule_risk * 0.4), 2)

    st.subheader(f"Fraud Risk Probability: {final_risk}")

    if final_risk > 0.7:
        st.error("⚠️ High Risk: Fraudulent Transaction")
    elif final_risk > 0.4:
        st.warning("⚠️ Medium Risk: Suspicious Transaction")
    else:
        st.success("✅ Low Risk: Transaction is Safe")
