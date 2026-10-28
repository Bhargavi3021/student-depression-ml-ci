
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

COPY app.py .
COPY student_depression_model.pkl .
COPY metrics.json .

EXPOSE 5000

CMD ["python", "app.py"]
