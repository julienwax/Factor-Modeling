FROM python:3.11-slim

RUN useradd --create-home appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Statistical_model/factor_utils.py .
COPY Statistical_model/run_model.py .

USER appuser
ENTRYPOINT ["python", "run_model.py"]
