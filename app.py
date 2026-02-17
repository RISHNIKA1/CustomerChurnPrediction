import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer will churn or not.")

model = joblib.load("models/model.pkl")

st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0, 200, 70)
TotalCharges = st.sidebar.slider("Total Charges", 0, 10000, 2000)


input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "InternetService": InternetService,
    "Contract": Contract,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_data])

expected_cols = model.named_steps["preprocessor"].feature_names_in_

for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0 if col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"] else "No"

input_df = input_df[expected_cols]

st.subheader("📌 Customer Input Data")
st.dataframe(input_df)

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer will CHURN (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {proba:.2f})")
