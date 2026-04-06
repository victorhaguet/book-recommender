# Indexing

This package builds and loads the retrieval index used by the application.

## Files

- `create_database.py`: creates the FAISS index from the ingested dataset.
- `embeddings.py`: selects and initializes the embedding backend.
- `faiss_store.py`: helpers for saving and loading the FAISS vector store.
- `retriever.py`: constructs the retriever used by the RAG layer.
