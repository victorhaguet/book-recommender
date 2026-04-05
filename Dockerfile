FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src
RUN pip install --no-cache-dir --no-deps .

FROM base AS backend

COPY conf /app/conf
COPY data /app/data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS frontend

COPY conf /app/conf

EXPOSE 8080

CMD ["chainlit", "run", "src/app_chainlit.py", "--host", "0.0.0.0", "--port", "8080"]
