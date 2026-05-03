import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# -------------------------
# LOAD MODELS
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))

# -------------------------
# TITLE
# -------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.write("AI-powered clinical decision support system")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("🧍 Patient Information")

age_group = st.sidebar.selectbox(
    "Age Group",
    ["20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
)

age_map = {
    "20-30": 25,
    "30-40": 35,
    "40-50": 45,
    "50-60": 55,
    "60-70": 65,
    "70+": 75
}
age = age_map[age_group]

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}
cp = st.sidebar.selectbox("Chest Pain Type", list(cp_map.keys()))
cp = cp_map[cp]

trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 200)

fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg_map = {
    "Normal": 0,
    "ST-T abnormality": 1,
    "LV hypertrophy": 2
}
restecg = st.sidebar.selectbox("Rest ECG", list(restecg_map.keys()))
restecg = restecg_map[restecg]

thalch = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

exang = st.sidebar.selectbox("Exercise Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)

slope_map = {
    "Upsloping (normal)": 0,
    "Flat (medium risk)": 1,
    "Downsloping (high risk)": 2
}
slope = st.sidebar.selectbox("Slope", list(slope_map.keys()))
slope = slope_map[slope]

ca_map = {
    "0 vessels": 0,
    "1 vessel": 1,
    "2 vessels": 2,
    "3 vessels": 3
}
ca = st.sidebar.selectbox("Major Vessels", list(ca_map.keys()))
ca = ca_map[ca]

thal_map = {
    "Normal": 0,
    "Fixed defect": 1,
    "Reversible defect": 2,
    "Unknown": 3
}
thal = st.sidebar.selectbox("Thal", list(thal_map.keys()))
thal = thal_map[thal]

# =========================
# PREDICTION
# =========================
if st.sidebar.button("Predict Risk"):

    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    data = pd.DataFrame([input_data])

    feature_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalch", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    data = data[feature_order]

    data = imputer.transform(data)
    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    risk = probability * 100

    # =========================
    # OUTPUT
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Score")
        st.progress(int(risk))
        st.metric("Risk Probability", f"{risk:.2f}%")

    with col2:
        st.subheader("🧠 Interpretation")

        if risk < 30:
            st.success("Low Risk")
        elif risk < 70:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

    st.info(f"Final model probability: {probability:.3f}")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    report = pd.DataFrame([{
        "Age": age,
        "Sex": "Male" if sex == 1 else "Female",
        "Chest Pain Type": cp,
        "Resting BP": trestbps,
        "Cholesterol": chol,
        "Fasting Blood Sugar": fbs,
        "Rest ECG": restecg,
        "Max Heart Rate": thalch,
        "Exercise Angina": exang,
        "Oldpeak": oldpeak,
        "Slope": slope,
        "Major Vessels": ca,
        "Thal": thal,
        "Risk Probability": round(probability, 4),
        "Risk Percentage": round(risk, 2)
    }])

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Patient Risk Report",
        data=csv,
        file_name="heart_disease_risk_report.csv",
        mime="text/csv"
    )