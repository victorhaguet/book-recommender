# Tests

This directory contains the automated test suite for the project.

## Structure

- `py_tests/`: Python tests grouped by application area.

This directory groups Python tests by feature area to mirror the application structure in `src/`.

## Subdirectories

- `chainlit_tests/`: frontend behavior and integration tests for the Chainlit app.
- `fastapi_tests/`: backend API tests for the FastAPI app.
- `indexing_tests/`: unit tests for embeddings, FAISS helpers, retriever creation, and index building.
- `ingestion_tests/`: tests for CSV loading and document preparation.
- `rag_tests/`: tests for prompts, formatting, chain behavior, and service orchestration.

## Scope

The tests cover ingestion, indexing, RAG logic, FastAPI behavior, and Chainlit integration.
