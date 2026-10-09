import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_prediction_endpoint(self):
        response = self.client.post(
            "/predict",
            json={
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
            }
        )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()

        self.assertEqual(
    result["prediction"],
    "INVALID"
)

        self.assertIn(
            result["prediction_code"],
            [0, 1]
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "Gender": "Female",
                "Age": 20
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
