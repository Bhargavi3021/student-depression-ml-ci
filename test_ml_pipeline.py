import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    # --------------------------------------------------
    # Test 1: Dataset Created / Available
    # --------------------------------------------------

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists("Student Depression Dataset.csv")
        )

    # --------------------------------------------------
    # Test 2: Model Created
    # --------------------------------------------------

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("student_depression_model.pkl")
        )

    # --------------------------------------------------
    # Test 3: Metrics Created
    # --------------------------------------------------

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    # --------------------------------------------------
    # Test 4: Accuracy Is Valid
    # --------------------------------------------------

    def test_accuracy_is_valid(self):

        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    # --------------------------------------------------
    # Test 5: Model Can Make Prediction
    # --------------------------------------------------

    def test_model_prediction(self):

        model = joblib.load(
            "student_depression_model.pkl"
        )

        sample = pd.DataFrame([{
            "Gender": "Female",
            "Age": 20,
            "City": "Kalyan",
            "Profession": "Student",
            "Academic Pressure": 3.0,
            "Work Pressure": 0.0,
            "CGPA": 8.0,
            "Study Satisfaction": 4.0,
            "Job Satisfaction": 0.0,
            "Sleep Duration": "7-8 hours",
            "Dietary Habits": "Healthy",
            "Degree": "B.Tech",
            "Have you ever had suicidal thoughts ?": "No",
            "Work/Study Hours": 6,
            "Financial Stress": 2.0,
            "Family History of Mental Illness": "No"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(int(prediction), [0, 1])

    # --------------------------------------------------
    # Test 6: Low-Risk Student Prediction
    # --------------------------------------------------

    def test_low_risk_student(self):

        model = joblib.load(
            "student_depression_model.pkl"
        )

        sample = pd.DataFrame([{
            "Gender": "Male",
            "Age": 21,
            "City": "Hyderabad",
            "Profession": "Student",
            "Academic Pressure": 1.0,
            "Work Pressure": 0.0,
            "CGPA": 9.0,
            "Study Satisfaction": 5.0,
            "Job Satisfaction": 0.0,
            "Sleep Duration": "7-8 hours",
            "Dietary Habits": "Healthy",
            "Degree": "B.Tech",
            "Have you ever had suicidal thoughts ?": "No",
            "Work/Study Hours": 4,
            "Financial Stress": 1.0,
            "Family History of Mental Illness": "No"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(int(prediction), [0, 1])

    # --------------------------------------------------
    # Test 7: Higher-Risk Student Prediction
    # --------------------------------------------------

    def test_higher_risk_student(self):

        model = joblib.load(
            "student_depression_model.pkl"
        )

        sample = pd.DataFrame([{
            "Gender": "Female",
            "Age": 22,
            "City": "Delhi",
            "Profession": "Student",
            "Academic Pressure": 5.0,
            "Work Pressure": 0.0,
            "CGPA": 5.5,
            "Study Satisfaction": 1.0,
            "Job Satisfaction": 0.0,
            "Sleep Duration": "Less than 5 hours",
            "Dietary Habits": "Unhealthy",
            "Degree": "B.Tech",
            "Have you ever had suicidal thoughts ?": "Yes",
            "Work/Study Hours": 12,
            "Financial Stress": 5.0,
            "Family History of Mental Illness": "Yes"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(int(prediction), [0, 1])


if __name__ == "__main__":
    unittest.main()
