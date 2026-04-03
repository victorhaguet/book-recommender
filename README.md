# Book RAG

`book-rag` is a retrieval-augmented generation project that recommends books from a catalog based on a natural-language query.

The repository is structured as a small production-style AI application:
- a FastAPI backend that builds or loads the retrieval stack and exposes the RAG endpoint
- a Chainlit frontend that provides a chat interface
- a modular indexing and retrieval pipeline built with LangChain and FAISS
- a Python test suite covering ingestion, indexing, RAG logic, FastAPI, and Chainlit integration

## Features

- Search books with natural-language requests such as `I want a fantasy book with trolls.`
- Build a FAISS vector index from a CSV dataset
- Load a prebuilt index to avoid rebuilding embeddings on every startup
- Support OpenAI-compatible embeddings or local Hugging Face embeddings
- Serve the backend and frontend locally or through Docker Compose
- Run a test suite from a single command

## Architecture

The application flow is:

1. A books dataset is loaded from CSV.
2. Each row is converted into a LangChain `Document`.
3. The `description` column becomes the text used for retrieval.
4. The remaining columns are stored as metadata.
5. Documents are embedded and stored in a FAISS vector store.
6. A retriever fetches the top `k` matching books for a user query.
7. Retrieved books are formatted into a prompt.
8. A chat model generates the final recommendation answer.

Main files:
- [app_fastapi.py](app_fastapi.py): FastAPI backend entrypoint
- [app_chainlit.py](app_chainlit.py): Chainlit frontend entrypoint
- [src/ingestion/load_books.py](src/ingestion/load_books.py): CSV loading and cleaning
- [src/indexing/create_database.py](src/indexing/create_database.py): index creation pipeline
- [src/indexing/embeddings.py](src/indexing/embeddings.py): embedding backend selection
- [src/indexing/retriever.py](src/indexing/retriever.py): retriever setup
- [src/rag/chain.py](src/rag/chain.py): core RAG chain
- [src/rag/rag_service.py](src/rag/rag_service.py): service orchestration

## Data

The repository includes a dataset at [data/books.csv](data/books.csv).

The ingestion layer assumes:
- the source is a CSV file
- `description` is used as the retrievable text
- all other kept columns become metadata attached to each document

By default, the example configuration drops:
- `isbn13`
- `isbn10`
- `thumbnail`

Rows with missing descriptions are removed during ingestion.

By default, the FAISS index is stored under [data/faiss_index](data/faiss_index).
If `FROM_SCRATCH=false`, the backend loads the existing index from that folder instead of rebuilding it.

## Requirements

- Python 3.11 recommended
- Docker and Docker Compose for containerized usage
- An OpenAI API key (or an OpenAI-compatible API key) if you use the default OpenAI embedding and chat configuration

## Configuration

Copy the example environment file and adapt it to your setup:

```bash
cp .env.example .env
```

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `STRATEGY` | Embeddings backend: `openai` or `hf` |
| `EMBEDDINGS_MODEL` | Embedding model name |
| `OPENAI_API_KEY` | API key for OpenAI-compatible services |
| `OPENAI_BASE_URL` | Optional base URL for OpenAI-compatible APIs |
| `FROM_SCRATCH` | `true` to rebuild the FAISS index, `false` to load an existing one |
| `DATA_PATH` | Path to the CSV dataset |
| `COLUMNS_TO_DROP` | Comma-separated list of columns removed before indexing |
| `FAISS_INDEX_PATH` | Directory containing FAISS files |
| `FAISS_INDEX_NAME` | Base name of the FAISS index (create it if it doesn't exist) |
| `CHAT_MODEL` | Chat model used for answer generation |
| `RETRIEVER_K` | Number of retrieved books per query |
| `RAG_ENDPOINT` | Backend endpoint used by Chainlit |

Default example values live in [.env.example](.env.example).

## Local Development

The Makefile is the recommended entrypoint for local usage.

Show available commands:

```bash
make help
```

Create a local virtual environment:

```bash
make env
source .venv/bin/activate
```

Install dependencies into the active environment:

```bash
make install
```

Run the test suite:

```bash
make test
```

Start the FastAPI backend:

```bash
make api
```

Start the Chainlit frontend:

```bash
make ui
```

Once both services are running:
- FastAPI is available on `http://127.0.0.1:8000`
- Chainlit is available on `http://127.0.0.1:8080`

## Docker Usage

Build and start the full stack:

```bash
make docker-up
```

Stop the stack:

```bash
make docker-down
```

The Compose setup starts:
- `backend` on port `8000`
- `frontend` on port `8080`

Both services are built from a single multi-stage [Dockerfile](Dockerfile):
- the `base` stage installs shared Python dependencies once
- the `backend` target runs FastAPI
- the `frontend` target runs Chainlit

By default, the backend reads the dataset and FAISS index from the `data/` directory. You can override these paths with environment variables if needed.

The frontend waits for the backend health check before starting.

## API Usage

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Query the RAG API:

```bash
curl -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query":"I want a fantasy book with dragons and political intrigue"}'
```

Expected response shape:

```json
{
  "response": "..."
}
```

## Frontend Usage

The Chainlit app is a lightweight chat client for the FastAPI backend.

When started, it:
- resolves the backend RAG endpoint from environment variables
- waits for the backend to be available
- forwards user messages to the `/rag` endpoint
- displays the generated recommendation text in the chat UI

## Testing

The project contains Python unit tests under [tests](tests).

Coverage includes:
- ingestion
- embeddings setup
- FAISS helpers
- retriever creation
- prompt and formatting logic
- RAG chain behavior
- FastAPI entrypoint behavior
- Chainlit entrypoint behavior

Run everything with:

```bash
make test
```

## Continuous Integration

The repository includes a GitHub Actions workflow at [.github/workflows/ci.yml](.github/workflows/ci.yml).

It runs automatically:
- on pushes to `main`
- on pull requests

The CI currently checks:
- Python dependency installation through `make install`
- the full Python test suite through `make test`
- Docker image builds for both the `backend` and `frontend` targets

You can view the result:
- in the repository `Actions` tab
- on each pull request through the status checks
- on commits through the GitHub check status icons

## Example Workflow

Local workflow:

```bash
cp .env.example .env
make env
source .venv/bin/activate
make install
make test
make api
```

In another terminal:

```bash
source .venv/bin/activate
make ui
```

Container workflow:

```bash
cp .env.example .env
make docker-up
```
