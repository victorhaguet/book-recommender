# Source Code

This directory contains the Python application code for the Book RAG project.

## Top-level modules

- `app_fastapi.py`: FastAPI entrypoint for the backend API.
- `app_chainlit.py`: Chainlit entrypoint for the chat frontend.
- `app_config.py`: Hydra-based configuration loader.
- `logging_utils.py`: shared logging helpers.

## Packages

- `ingestion/`: dataset loading and preprocessing.
- `indexing/`: embedding, FAISS, and retriever setup.
- `rag/`: prompt formatting, chain construction, and service orchestration.
