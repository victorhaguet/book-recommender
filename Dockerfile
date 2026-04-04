FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

FROM base AS backend

COPY app_fastapi.py /app/app_fastapi.py
COPY conf /app/conf
COPY src /app/src
COPY data /app/data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS frontend

COPY app_chainlit.py /app/app_chainlit.py
COPY conf /app/conf
COPY src /app/src

EXPOSE 8080

CMD ["chainlit", "run", "app_chainlit.py", "--host", "0.0.0.0", "--port", "8080"]
