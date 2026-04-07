# RAG

This package contains the retrieval-augmented generation logic.

## Files

- `chain.py`: builds the core RAG chain.
- `formatting.py`: formats retrieved book data for prompt input or output rendering.
- `prompts.py`: loads prompt templates and builds LangChain prompt objects.
- `templates/default_generation_prompt.jinja2`: default Jinja prompt used for answer generation.
- `rag_service.py`: orchestrates retrieval and generation for the application entrypoints.
