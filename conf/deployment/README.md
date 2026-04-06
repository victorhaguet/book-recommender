# Deployment Configurations

This directory contains small Hydra overrides for deployment-specific frontend addresses.

## Files

- `local.yaml`: points the frontend to the local FastAPI endpoint at `http://127.0.0.1:8000/rag`.
- `docker.yaml`: points the frontend to the Docker Compose backend service at `http://backend:8000/rag`.

## Usage

Set `APP_ENV` to `local` or `docker` to load the matching endpoint override at startup.
