import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------
# LOAD MODELS
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))

# -------------------------
# UI TITLE
# -------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.write("AI-powered medical risk analysis system")

# -------------------------
# SMART INPUT UI
# -------------------------
input_dict = {}

with st.form("form"):

    st.subheader("🧍 Patient Information")

    col1, col2 = st.columns(2)

    # AGE GROUP
    with col1:
        age_group = st.selectbox(
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

        input_dict["age"] = age_map[age_group]

    # SEX
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        input_dict["sex"] = 1 if sex == "Male" else 0

    # CHEST PAIN TYPE
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }

    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    input_dict["cp"] = cp_map[cp]

    # NUMERIC FEATURES
    input_dict["trestbps"] = st.slider("Resting Blood Pressure", 80, 200, 120)
    input_dict["chol"] = st.slider("Cholesterol", 100, 600, 200)

    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
    input_dict["fbs"] = 1 if fbs == "Yes" else 0

    # -------------------------
    # REST ECG (FIXED)
    # -------------------------
    restecg_map = {
        "0 - Normal": 0,
        "1 - ST-T wave abnormality": 1,
        "2 - Left ventricular hypertrophy": 2
    }

    restecg = st.selectbox("Rest ECG", list(restecg_map.keys()))
    input_dict["restecg"] = restecg_map[restecg]

    # HEART RATE
    input_dict["thalch"] = st.slider("Max Heart Rate", 60, 220, 150)

    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    input_dict["exang"] = 1 if exang == "Yes" else 0

    input_dict["oldpeak"] = st.slider("Oldpeak", 0.0, 6.0, 1.0)

    # -------------------------
    # SLOPE (FIXED)
    # -------------------------
    slope_map = {
        "0 - Upsloping (Normal)": 0,
        "1 - Flat (Medium risk)": 1,
        "2 - Downsloping (High risk)": 2
    }

    slope = st.selectbox("Slope of ST segment", list(slope_map.keys()))
    input_dict["slope"] = slope_map[slope]

    # -------------------------
    # CA (Vessels)
    # -------------------------
    ca_map = {
        "0 - No major vessel blocked": 0,
        "1 - 1 vessel blocked": 1,
        "2 - 2 vessels blocked": 2,
        "3 - 3 vessels blocked": 3
    }

    ca = st.selectbox("Number of major vessels colored", list(ca_map.keys()))
    input_dict["ca"] = ca_map[ca]

    # -------------------------
    # THAL (FIXED)
    # -------------------------
    thal_map = {
        "0 - Normal": 0,
        "1 - Fixed defect": 1,
        "2 - Reversible defect": 2,
        "3 - Unknown": 3
    }

    thal = st.selectbox("Thal (stress test result)", list(thal_map.keys()))
    input_dict["thal"] = thal_map[thal]

    submitted = st.form_submit_button("Predict Risk")

# -------------------------
# PREDICTION
# -------------------------
if submitted:

    data = pd.DataFrame([input_dict])

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

    st.subheader("📊 Risk Result")

    st.progress(int(risk))
    st.metric("Heart Disease Risk", f"{risk:.2f}%")

    if risk < 30:
        st.success("🟢 Low Risk")
        st.write("No major cardiovascular risk detected.")

    elif risk < 70:
        st.warning("🟡 Moderate Risk")
        st.write("Some risk factors present. Monitoring recommended.")

    else:
        st.error("🔴 High Risk")
        st.write("High probability detected. Medical attention recommended.")

    st.write(f"Probability score: **{probability:.2f}**")