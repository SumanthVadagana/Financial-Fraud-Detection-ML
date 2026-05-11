import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Financial Fraud Detection System")

st.write("Enter transaction details below")

# Inputs
step = st.number_input("Step")

type_value = st.selectbox(
    "Transaction Type",
    ["CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT", "CASH_IN"]
)

# Encoding map
type_mapping = {
    "CASH_OUT": 0,
    "PAYMENT": 1,
    "TRANSFER": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}

type_encoded = type_mapping[type_value]

amount = st.number_input("Amount")

oldbalanceOrg = st.number_input("Old Balance Origin")

newbalanceOrig = st.number_input("New Balance Origin")

oldbalanceDest = st.number_input("Old Balance Destination")

newbalanceDest = st.number_input("New Balance Destination")

# Predict
if st.button("Predict"):

    input_data = pd.DataFrame([[
        step,
        type_encoded,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    ]], columns=[
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Legitimate Transaction")