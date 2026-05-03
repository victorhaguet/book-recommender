![Book recommender](/public/book-recommender.png)

# Book Recommender

`book-recommender` is an agentic retrieval-augmented generation project that recommends books from a catalog and supports follow-up discussion about recommended books and authors.

The repository is structured as a small production-style AI application:
- a FastAPI backend that builds or loads the retrieval stack, hosts the LangGraph workflow, and exposes the RAG endpoint
- a Chainlit frontend that provides a chat interface
- a modular indexing and retrieval pipeline built with LangChain and FAISS
- a Python test suite covering ingestion, indexing, prompts, RAG logic, agentic routing, FastAPI, and Chainlit integration

## Features

- Search books with natural-language recommendation requests such as `I want a fantasy book with trolls.`
- Separate query recognition from recommendation generation
- Maintain per-thread conversation context through a backend-managed `thread_id`
- Route turns between recommendation, follow-up, clarification, and rejection paths with LangGraph
- Run Tavily, Wikipedia, and LLM background lookups in parallel for follow-up questions
- Return structured recommendation cards and follow-up source links
- Build a FAISS vector index from a CSV dataset
- Load a prebuilt index to avoid rebuilding embeddings on every startup
- Support OpenAI-compatible embeddings or local Hugging Face embeddings
- Configure separate chat models for recommendation generation and query recognition
- Serve the backend and frontend locally or through Docker Compose
- Run the full test suite and coverage checks from single commands

## Architecture

The application has two layers:

1. An indexing layer that loads `data/books.csv`, converts rows into LangChain `Document` objects, embeds them, and stores them in FAISS.
2. An agentic runtime layer that uses LangGraph to decide whether a user turn should:
   - generate new book recommendations from the catalog
   - answer a follow-up question about recommended books or authors
   - ask for clarification
   - reject the request as out of scope

### Agentic Workflow

![Agentic workflow graph](/public/agentic-workflow.png)

### Runtime Flow

1. Chainlit creates or reuses a `thread_id` for the chat session.
2. FastAPI forwards the request to `AgenticRAGService`.
3. LangGraph classifies the turn before any generation happens.
4. For recommendation turns, the existing FAISS retriever and recommendation prompt are used.
5. For follow-up turns, the graph fans out into Tavily, Wikipedia, and LLM background nodes in parallel, then synthesizes a final answer with sources.
6. The backend updates thread memory so the next user turn can use prior recommendations and recent conversation context.

Main files:
- [src/app_fastapi.py](src/app_fastapi.py): FastAPI backend entrypoint
- [src/app_chainlit.py](src/app_chainlit.py): Chainlit frontend entrypoint
- [src/ingestion/load_books.py](src/ingestion/load_books.py): CSV loading and cleaning
- [src/indexing/create_database.py](src/indexing/create_database.py): index creation pipeline
- [src/indexing/embeddings.py](src/indexing/embeddings.py): embedding backend selection
- [src/indexing/retriever.py](src/indexing/retriever.py): retriever setup
- [src/rag/chain.py](src/rag/chain.py): core RAG chain
- [src/rag/rag_service.py](src/rag/rag_service.py): service orchestration
- [src/rag/agentic_service.py](src/rag/agentic_service.py): LangGraph routing, thread memory, and follow-up orchestration
- [src/rag/prompts.py](src/rag/prompts.py): prompt/template loading helpers

## Data

The repository includes a dataset at [data/books.csv](data/books.csv).

The ingestion layer assumes:
- the source is a CSV file
- rows with blank or very short descriptions are excluded from indexing
- the retrievable text is built from `title`, `authors`, `categories`, and `description`
- all other kept columns become metadata attached to each document

By default, the example configuration drops:
- `isbn13`
- `isbn10`

By default, descriptions shorter than `10` words are removed during ingestion.

By default, the FAISS index is stored under [data/faiss_index](data/faiss_index).
If `index.from_scratch: false`, the backend loads the existing index from that folder instead of rebuilding it.

## Requirements

- Python 3.11 recommended
- Docker and Docker Compose for containerized usage
- An OpenAI API key (or an OpenAI-compatible API key) if you use the default OpenAI embedding and chat configuration

## Configuration

Copy the example environment file and adapt it to your setup:

```bash
cp .env.example .env
```

Application settings are now stored in Hydra config files:

- [conf/config.yaml](conf/config.yaml): shared defaults for embeddings, LLM, indexing, RAG, and frontend settings
- [conf/deployment/local.yaml](conf/deployment/local.yaml): local frontend endpoint
- [conf/deployment/docker.yaml](conf/deployment/docker.yaml): Docker frontend endpoint

The `.env` file is only for secrets and private runtime values.

Important secret environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Optional shared fallback key for OpenAI-compatible services |
| `APP_ENV` | Optional deployment profile selector. Defaults to `local`; Docker uses `docker` |

Default example values live in [.env.example](.env.example).

Example Hydra settings live in [conf/config.yaml](conf/config.yaml). To change the embedding backend, LLM model, classifier model, retriever `k`, follow-up search limits, index paths, or frontend endpoint, update the config files.

Agentic settings in [conf/config.yaml](conf/config.yaml):
- `agentic.classifier_model`: optional dedicated model for query recognition
- `agentic.tavily_max_results`: maximum Tavily results for follow-up questions
- `agentic.wikipedia_top_k_results`: maximum Wikipedia results for follow-up questions

Important extra secrets for the agentic follow-up flow:

| Variable | Purpose |
| --- | --- |
| `TAVILY_API_KEY` | Enables Tavily search in the follow-up branch |

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

Install the optional local Hugging Face embedding stack only if you set `embeddings.strategy: hf`:

```bash
make install-hf
```

Run the test suite:

```bash
make test
```

Run the test suite with coverage enforcement:

```bash
make coverage
```

Generate an HTML coverage report in `htmlcov/`:

```bash
make coverage-html
```

Show missing coverage lines for a single source file:

```bash
make coverage-file FILE=src/rag/prompts.py
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
- the `base` stage installs pinned third-party Python dependencies from `requirements.lock`
- the application package is then installed separately from the local source with `pip install --no-deps .`
- the `backend` target runs FastAPI
- the `frontend` target runs Chainlit

By default, the backend reads the dataset and FAISS index from the `data/` directory. You can override these paths with environment variables if needed.
By default, the backend reads these values from Hydra config instead of environment variables.

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
  -d '{"thread_id":"demo-thread","query":"I want a fantasy book with dragons and political intrigue"}'
```

Expected response shape:

```json
{
  "response": "...",
  "sources": [
    {
      "title": "...",
      "url": "..."
    }
  ],
  "recommendations": [
    {
      "title": "...",
      "thumbnail": "...",
      "author": "...",
      "summary": "...",
      "num_pages": 123
    }
  ]
}
```

Notes:
- `thread_id` is required so the backend can keep per-conversation memory.
- Recommendation turns usually return populated `recommendations` and an empty `sources` list.
- Follow-up turns usually return an empty `recommendations` list and a populated `sources` list.

## Frontend Usage

The Chainlit app is a lightweight chat client for the FastAPI backend.

When started, it:
- resolves the backend RAG endpoint from Hydra config
- waits for the backend to be available
- creates a session-level `thread_id`
- forwards user messages to the `/rag` endpoint with that `thread_id`
- displays recommendation turns as an introduction followed by structured recommendation cards
- displays follow-up turns as sourced prose answers

Each recommendation card is rendered from structured backend data instead of relying only on raw LLM prose. Follow-up answers can additionally render a `Sources` section from the backend response.

### Example Result

The image below shows a visual example of a recommendation result in the Chainlit interface for a specific query.

![Visual example of a book recommendation result](/public/book-recommender-visual.png)

## Testing

The project contains Python unit tests under [tests](tests).

Coverage includes:
- ingestion
- embeddings setup
- FAISS helpers
- retriever creation
- prompt and formatting logic
- RAG chain behavior
- agentic routing and thread handling
- FastAPI entrypoint behavior
- Chainlit entrypoint behavior

Run everything with:

```bash
make test
```

Run the full suite with coverage for `src/` and fail below `90%` total coverage:

```bash
make coverage
```

Generate an HTML report you can inspect in the browser:

```bash
make coverage-html
```

Show missing line numbers for a specific file:

```bash
make coverage-file FILE=src/rag/prompts.py
```

## Continuous Integration

The repository includes a GitHub Actions workflow at [.github/workflows/ci.yml](.github/workflows/ci.yml).

It runs automatically:
- on pushes to `main`
- on pull requests

The CI currently checks:
- Python dependency installation through `make install-hf`
- coverage for the `src/` package through `make coverage`
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
