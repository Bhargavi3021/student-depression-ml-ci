from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("student_depression_model.pkl")

FEATURES = [
    "Gender",
    "Age",
    "City",
    "Profession",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_depression_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)



@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-depression-prediction"
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "student-depression-prediction"
    })



@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    model = load_model()
    prediction_code = int(model.predict(sample)[0])

    prediction = (
        "DEPRESSED"
        if prediction_code == 1
        else "NOT DEPRESSED"
    )

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
