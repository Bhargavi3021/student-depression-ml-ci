
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "student_depression_model.pkl"

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger("student_depression")

# Prometheus metrics
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Depression API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Depression prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Depression model loaded successfully"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_depression_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


# Load the model and update readiness metric
try:
    model = load_model()
    MODEL_LOADED.set(1)
    logger.info("event=model_loaded model_file=%s", MODEL_PATH.name)
except Exception:
    MODEL_LOADED.set(0)
    logger.exception("event=model_load_failed model_file=%s", MODEL_PATH.name)
    raise


# Log requests and record HTTP metrics
@app.before_request
def before_request():
    request.request_id = str(uuid.uuid4())
    request.start_time = time.perf_counter()

    logger.info(
        "event=request_started request_id=%s method=%s path=%s",
        request.request_id,
        request.method,
        request.path
    )


@app.after_request
def after_request(response):
    duration_seconds = (
        time.perf_counter() -
        getattr(request, "start_time", time.perf_counter())
    )
    duration_ms = round(duration_seconds * 1000, 2)

    request_id = getattr(request, "request_id", "unknown")

    logger.info(
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    # Exclude metric scraping from user traffic metrics
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_seconds)

    response.headers["X-Request-ID"] = request_id
    return response


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


@app.get("/metrics")
def metrics():
    return app.response_class(
        response=generate_latest(),
        status=200,
        mimetype=CONTENT_TYPE_LATEST
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning(
            "event=validation_error request_id=%s reason=missing_or_invalid_json",
            request.request_id
        )
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error request_id=%s missing_fields=%s",
            request.request_id,
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    logger.info("event=prediction_started request_id=%s", request.request_id)

    try:
        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "DEPRESSED"
            if prediction_code == 1
            else "NOT DEPRESSED"
        )

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success request_id=%s prediction_code=%s",
            request.request_id,
            prediction_code
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            request.request_id
        )
        return jsonify({
            "error": "Prediction failed",
            "request_id": request.request_id
        }), 500


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return error

    logger.exception(
        "event=application_error request_id=%s",
        getattr(request, "request_id", "unknown")
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(request, "request_id", "unknown")
    }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
