import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model/predictive_model.pkl")

st.title("🔧 AI Predictive Maintenance System")

st.write("Enter machine parameters below:")

# Inputs
machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

machine_type = {
    "L": 0,
    "M": 1,
    "H": 2
}[machine_type]

air_temp = st.number_input("Air Temperature (K)", value=300.0)

process_temp = st.number_input("Process Temperature (K)", value=310.0)

rpm = st.number_input("Rotational Speed (RPM)", value=1500)

torque = st.number_input("Torque (Nm)", value=40.0)

tool_wear = st.number_input("Tool Wear (min)", value=50)

# Prediction Button

if st.button("Predict"):

    data = pd.DataFrame([[
    machine_type,
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear
    ]], columns=[
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
    ])
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    failure_prob = probability[0][1] * 100
    health_score = 100 - failure_prob

    if prediction[0] == 1:
        st.error("⚠️ Machine Failure Expected")
    else:
        st.success("✅ Machine Healthy")

    st.metric(
        "Machine Health Score",
        f"{health_score:.2f}%"
    )

    st.write(f"Failure Probability: {failure_prob:.2f}%")

    # Feature Importance Graph
    feature_names = [
    "Type",
    "Air Temp",
    "Process Temp",
    "RPM",
    "Torque",
    "Tool Wear"
    ]

    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, importance)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance Analysis")

    st.subheader("Feature Importance")
    st.pyplot(fig)