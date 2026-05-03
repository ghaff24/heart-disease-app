#  Heart Disease Risk Predictor (Machine Learning Web App)

An interactive Streamlit web application that predicts the risk of heart disease based on patient medical attributes using a trained Logistic Regression machine learning model.

##  Live Demo
(https://heart-disease-app-ghaff24.streamlit.app/)

---

##  Project Overview
This project uses a supervised machine learning model trained on the Heart Disease UCI dataset to predict whether a patient is at low, moderate, or high risk of heart disease.

The app provides:
- Interactive UI for entering patient data
- Real-time prediction
- Risk probability score
- Human-readable interpretation
- Downloadable patient report (CSV)

---

##  Features Used
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Rest ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak (ST depression)
- Slope of ST segment
- Number of major vessels (ca)
- Thalassemia result

---

## Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Machine Learning: Logistic Regression

---

##  Model Workflow
1. Data cleaning & preprocessing
2. Handling missing values (SimpleImputer)
3. Feature scaling (StandardScaler)
4. Model training (Logistic Regression)
5. Model serialization (pickle)
6. Deployment with Streamlit

---

##  Files in Repository
- `app.py` → Streamlit web app
- `model.pkl` → trained ML model
- `scaler.pkl` → feature scaler
- `imputer.pkl` → missing value handler
- `heart_disease_uci.csv` → dataset
- `requirements.txt` → dependencies

pip install -r requirements.txt
streamlit run app.py
