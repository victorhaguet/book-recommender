# Ingestion

This package is responsible for loading the book dataset and converting it into the structures used by the retrieval pipeline.

## Files

- `load_books.py`: reads the CSV source, drops configured columns, filters invalid rows, and builds LangChain documents with metadata.
