"""Diabetic Prediction Model - Streamlit app.

Dataset: Kaggle "iammustafatz/diabetes-prediction-dataset"
Columns: gender, age, hypertension, heart_disease, smoking_history,
         bmi, HbA1c_level, blood_glucose_level, diabetes (target)

Run with:
    streamlit run app.py

Put `diabetes_prediction_dataset.csv` in the same folder as this file.
On first run the app trains a model, saves it as `diabetes_pipeline.pkl`,
and reuses it on later runs.
"""

import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "diabetes_prediction_dataset.csv"
MODEL_PATH = "diabetes_pipeline.pkl"

NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_history"]
FEATURES = ["gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

GENDERS = ["Female", "Male", "Other"]
SMOKING = ["never", "No Info", "current", "former", "ever", "not current"]


def build_pipeline():
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocess),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    class_weight="balanced",  # dataset is heavily imbalanced
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model():
    if not os.path.exists(DATA_PATH):
        return None, None
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X, y = df[FEATURES], df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "recall": recall_score(y_test, pred),
    }
    joblib.dump(model, MODEL_PATH)
    return model, metrics


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), None
    return train_model()


def main():
    st.set_page_config(page_title="Diabetic Prediction", page_icon="🩺", layout="centered")
    st.title("🩺 Diabetic Prediction Model")
    st.write("Enter the patient's details to estimate the likelihood of diabetes.")

    model, metrics = load_model()
    if model is None:
        st.error(
            f"Could not find `{DATA_PATH}` or `{MODEL_PATH}` in this folder. "
            "Download the dataset from Kaggle (iammustafatz/diabetes-prediction-dataset), "
            "place the CSV next to app.py, and reload."
        )
        st.stop()

    if metrics:
        st.info(
            f"Model trained now. Test accuracy: {metrics['accuracy']:.2%} | "
            f"Recall (diabetic class): {metrics['recall']:.2%}"
        )

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", GENDERS)
        age = st.number_input("Age", 1, 120, 40)
        bmi = st.number_input("BMI", 10.0, 100.0, 25.0, step=0.1)
        smoking = st.selectbox("Smoking history", SMOKING)
    with col2:
        hba1c = st.number_input("HbA1c level (%)", 3.0, 15.0, 5.5, step=0.1)
        glucose = st.number_input("Blood glucose level (mg/dL)", 50, 400, 120)
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart disease", ["No", "Yes"], horizontal=True)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame(
            [{
                "gender": gender,
                "age": age,
                "hypertension": int(hypertension == "Yes"),
                "heart_disease": int(heart_disease == "Yes"),
                "smoking_history": smoking,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
            }],
            columns=FEATURES,
        )
        prob = float(model.predict_proba(row)[0][1])

        st.subheader("Result")
        if prob >= 0.5:
            st.error(f"High risk of diabetes (probability: {prob:.1%})")
        else:
            st.success(f"Low risk of diabetes (probability: {prob:.1%})")
        st.progress(prob)

    st.caption(
        "⚠️ Educational use only. This is not a medical diagnosis. "
        "Please consult a doctor for medical advice."
    )


if __name__ == "__main__":
    main()
