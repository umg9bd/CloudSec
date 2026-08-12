# CPU image for P_seq realtime scorer
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-prod.txt .
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

COPY prod/ ./prod/
COPY artifacts/temporal_lstm_v2.pt ./artifacts/temporal_lstm_v2.pt

EXPOSE 8000

CMD ["uvicorn", "prod.app:app", "--host", "0.0.0.0", "--port", "8000"]
