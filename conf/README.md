# Configuration

This directory stores the Hydra configuration used by the application.

## Files

- `config.yaml`: shared defaults for embeddings, indexing, RAG, and frontend settings.

## Subdirectories

- `deployment/`: environment-specific overrides that select runtime endpoints and deployment behavior.

## Usage

Application settings are loaded through `src/app_config.py`. The active deployment profile is selected with the `APP_ENV` environment variable and defaults to `local`.
