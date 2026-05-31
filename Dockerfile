FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Statistical_model/factor_utils.py .
COPY Statistical_model/run_model.py .

ENTRYPOINT ["python", "run_model.py"]
