from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

# PDF file name
pdf_file = "Credit_Card_Fraud_Prediction_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
styles = getSampleStyleSheet()
story = []

# Custom styles
title_style = styles['Title']
heading_style = styles['Heading2']
normal_style = styles['Normal']
code_style = ParagraphStyle(name='Code', fontName='Courier', fontSize=8, leading=12, backColor=colors.lightgrey)

# 1️⃣ Front Page
story.append(Paragraph("💳 Credit Card Fraud Prediction System", title_style))
story.append(Spacer(1, 20))
story.append(Paragraph("Author: Zishan Khan", normal_style))
story.append(Paragraph("Date: January 2026", normal_style))
story.append(Spacer(1, 20))
story.append(Paragraph("This project predicts the fraud risk of credit card transactions using a hybrid ML + Rule-based system.", normal_style))
story.append(Spacer(1, 40))

# App Screenshot Placeholder (replace 'app_front.png' with your screenshot)
try:
    story.append(Image("app_front.png", width=400, height=300))
    story.append(Spacer(1, 20))
except:
    story.append(Paragraph("[App Front Page Screenshot Here]", normal_style))
    story.append(Spacer(1, 20))

# 2️⃣ Abstract
story.append(Paragraph("Abstract", heading_style))
abstract_text = """
This project predicts the fraud risk of credit card transactions. It combines a Machine Learning model trained on realistic synthetic data with a rule-based engine. 
The system provides users with a risk assessment while clearly noting that it does not confirm actual fraud. 
The app is user-friendly and can be accessed from both laptop and mobile devices.
"""
story.append(Paragraph(abstract_text, normal_style))
story.append(Spacer(1, 10))

# 3️⃣ Introduction
story.append(Paragraph("Introduction", heading_style))
intro_text = """
Credit card fraud is a major issue globally. Detecting potential fraudulent transactions quickly can reduce financial loss.
The objective of this project is to build a system that allows users to input transaction details and receive a fraud risk prediction.
"""
story.append(Paragraph(intro_text, normal_style))
story.append(Spacer(1, 10))

# 4️⃣ Technology Stack
story.append(Paragraph("Technology Stack", heading_style))
tech_text = """
- Frontend: Streamlit (Python)
- Machine Learning: scikit-learn, Logistic Regression
- Rule Engine: Custom Python logic (rules.py)
- Data Handling: pandas, numpy
- Environment: Python 3.12
"""
story.append(Paragraph(tech_text, normal_style))
story.append(Spacer(1, 10))

# 5️⃣ Project Structure
story.append(Paragraph("Project Structure", heading_style))
structure_text = """
credit_card_fraud_detection/
├── app.py                 # Streamlit frontend
├── train_model.py         # ML model training
├── model.pkl              # Saved ML model
├── rules.py               # Rule-based logic
├── data_generator.py      # Synthetic realistic data
└── requirements.txt       # Dependencies
"""
story.append(Paragraph(structure_text, code_style))
story.append(Spacer(1, 10))

# 6️⃣ ML Model Training Code
story.append(Paragraph("Machine Learning Model Training (train_model.py)", heading_style))
ml_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv("transactions.csv")

X = df.drop("fraud", axis=1)
y = df["fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
"""
story.append(Paragraph(ml_code, code_style))
story.append(Spacer(1, 10))

# 7️⃣ Streamlit Frontend Code
story.append(Paragraph("Streamlit Frontend (app.py)", heading_style))
frontend_code = """
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

# User inputs
amount = st.number_input("Transaction Amount (₹)", min_value=1.0, step=100.0)
hour = st.slider("Transaction Hour (0–23)", 0, 23, 12)
location_change = st.selectbox("Transaction from new city?", ["No", "Yes"])
device_change = st.selectbox("Transaction from new device?", ["No", "Yes"])
txn_count_24h = st.slider("Transactions in last 24 hours", 1, 20, 1)

# Convert Yes/No to 0/1
location_change = 1 if location_change == "Yes" else 0
device_change = 1 if device_change == "Yes" else 0

if st.button("Check Fraud Risk"):
    input_data = np.array([[amount, hour, location_change, device_change, txn_count_24h]])
    ml_prob = model.predict_proba(input_data)[0][1]
    rule_risk = rule_based_check(amount, hour, location_change, device_change, txn_count_24h)
    final_risk = round((ml_prob * 0.6 + rule_risk * 0.4), 2)

    st.subheader(f"Fraud Risk Probability: {final_risk}")

    if final_risk > 0.7:
        st.error("⚠️ High Risk: Fraudulent Transaction")
    elif final_risk > 0.4:
        st.warning("⚠️ Medium Risk: Suspicious Transaction")
    else:
        st.success("✅ Low Risk: Transaction is Safe")
"""
story.append(Paragraph(frontend_code, code_style))
story.append(Spacer(1, 10))

# 8️⃣ Demo / Usage
story.append(Paragraph("Demo / Usage Instructions", heading_style))
demo_text = """
1. Run the app: streamlit run app.py
2. Enter transaction details (amount, hour, new city/device, transactions count)
3. Click 'Check Fraud Risk'
4. Observe output:
   - Low Risk → ✅ Safe
   - Medium Risk → ⚠️ Suspicious
   - High Risk → 🚨 Fraud
"""
story.append(Paragraph(demo_text, normal_style))
story.append(Spacer(1, 10))

# 9️⃣ Conclusion
story.append(Paragraph("Conclusion", heading_style))
conclusion_text = """
- Hybrid ML + Rule-based system provides realistic fraud risk prediction
- User-friendly interface
- Limitation: Only risk prediction, not actual fraud confirmation
"""
story.append(Paragraph(conclusion_text, normal_style))
story.append(Spacer(1, 10))

# 10️⃣ References
story.append(Paragraph("References", heading_style))
refs_text = """
- Python Documentation
- Streamlit Documentation
- scikit-learn Documentation
- Synthetic credit card fraud datasets research
"""
story.append(Paragraph(refs_text, normal_style))

# Build PDF
doc.build(story)

print(f"PDF report generated: {pdf_file}")
