"""
This module contain the FastAPI application entrypoint for the RAG service.
It mirrors the Chainlit setup: the vectorstore is built/loaded on startup
and the RAG endpoint is unavailable until initialization completes.

Start to run the Fast API app : 
python3 -m uvicorn app_fastapi:app --reload

Query the RAG service:
curl -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"<your-query>\"}"
"""

import os
from typing import Optional, overload, Literal, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.indexing.embeddings import get_embeddings
from src.indexing.faiss_store import load_store
from src.indexing.create_database import create_database
from src.rag.rag_service import RAGService

load_dotenv()

app = FastAPI()

_RAG_INSTANCE: Optional[RAGService] = None
_RAG_ERROR: Optional[Exception] = None


@overload
def _get_env(keys: list[str], default: str, required: Literal[True, False] = False) -> str: ...


@overload
def _get_env(keys: list[str], default: None = None, required: Literal[True] = True) -> str: ...


@overload
def _get_env(keys: list[str], default: Optional[str] = None, required: Literal[False] = False) -> str | None: ...


@overload
def _get_env(keys: list[str], default: Optional[str] = None, required: bool = False) -> str | None: ...


def _get_env(keys: list[str], default: Optional[str] = None, required: bool = False) -> str | None:
    """
    Get environment variable from a list of possible keys.

    :param keys: List of possible environment variable keys
    :type keys: list[str]
    :param default: Default value if no environment variable is found
    :type default: Optional[str]
    :param required: Whether at least one environment variable is required
    :type required: bool
    :return: The value of the first found environment variable or the default value
    :rtype: str | None
    """
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    if required and default is None:
        raise ValueError(f"Missing environment variable: {keys[0]}")
    return default

def get_env(keys: list[str], default: Optional[str] = None, required: bool = False) -> str | None:
    """
    Public wrapper for environment lookup (exposed for tests/consumers).
    
    :param keys: List of possible environment variable keys
    :type keys: list[str]
    :param default: Default value if no environment variable is found
    :type default: Optional[str]
    :param required: Whether at least one environment variable is required
    :type required: bool
    :return: The value of the first found environment variable or the default value
    :rtype: str | None
    
    """
    return _get_env(keys, default, required)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """
    Parse common truthy/falsey strings into a boolean.
    
    :param value: The string to be parsed into a boolean.
    :type value: str | None
    :param default: The default value returned if the input string is invalid.
    :type default: bool
    :return: The parsed boolean value.
    :rtype: bool
    """
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_csv_list(value: str | None, default: Optional[List[str]] = None) -> List[str]:
    """
    Parse a comma-separated string into a list.

    :param value: A comma-separated string of values.
    :type value: str | None
    :param default: The default value to return if the input is None.
    :type default: Optional[List[str]]
    :return: A list of strings.
    :rtype: List[str]
    
    """
    if value is None:
        return default if default is not None else []
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _init_rag() -> RAGService:
    """
    Initialize the RAG service from environment variables.

    :return: Initialized RAG service
    :rtype: RAGService
    """
    strategy: str = _get_env(["STRATEGY", "strategy"], None, True)
    embeddings_model: str = _get_env(["EMBEDDINGS_MODEL", "EMBEDDING_MODEL", "model"], None, True)
    api_key: str = _get_env(["OPENAI_API_KEY", "api_key"], None, True)
    base_url: Optional[str] = _get_env(["OPENAI_BASE_URL", "base_url"], None, False)

    embeddings: Embeddings = get_embeddings(
        strategy=strategy,
        model=embeddings_model,
        api_key=api_key,
        base_url=base_url,
    )

    from_scratch: bool = _parse_bool(
        _get_env(["FROM_SCRATCH", "from_scratch"], None, False),
        default=True,
    )
    index_path: str = _get_env(["FAISS_INDEX_PATH", "INDEX_PATH"], "/data/faiss_index/", True)
    index_name: str = _get_env(["FAISS_INDEX_NAME", "INDEX_NAME"], "books_index", True)
    data_path: str = _get_env(["DATA_PATH", "data_path"], "/data/books.csv", False)
    columns_to_drop: List[str] = _parse_csv_list(
        _get_env(["COLUMNS_TO_DROP", "columns_to_drop"], None, False),
        default=[],
    )

    vectorstore: FAISS
    if from_scratch:
        vectorstore = create_database(
            data_path=data_path,
            columns_to_drop=columns_to_drop,
            embeddings=embeddings,
            index_path=index_path,
            index_name=index_name,
        )
    else:
        vectorstore = load_store(embeddings=embeddings, path=index_path, index_name=index_name)

    chat_model = _get_env(["CHAT_MODEL", "LLM_MODEL"], "gpt-3.5-turbo", True)
    llm = ChatOpenAI(model=chat_model, api_key=SecretStr(api_key), base_url=base_url, temperature=0)
    k = int(_get_env(["RETRIEVER_K", "K"], "5"))

    return RAGService(llm=llm, vectorstore=vectorstore, k=k)


@app.on_event("startup")
def on_app_startup() -> None:
    """
    Build/load the vectorstore once at app startup so the endpoint is blocked until ready.
    """
    global _RAG_INSTANCE, _RAG_ERROR
    try:
        _RAG_INSTANCE = _init_rag()
        _RAG_ERROR = None
    except Exception as exc:
        _RAG_INSTANCE = None
        _RAG_ERROR = exc


class RagRequest(BaseModel):
    query: str


class RagResponse(BaseModel):
    response: str


@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(payload: RagRequest) -> RagResponse:
    """
    Call the endpoint with a query (string) and obtain 
    the RAG response.
    
    :param payload: User request
    :type payload: RagRequest
    :return: Description
    :rtype: RagResponse
    """
    # Check if it runs properly
    if _RAG_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"RAG startup error: {_RAG_ERROR}")

    # Check if the vectorstore is ready
    if _RAG_INSTANCE is None:
        raise HTTPException(status_code=503, detail="We are currently learning the books, please wait.")

    # Try to call the RAG instance
    try:
        result = _RAG_INSTANCE.answer_query(payload.query)
        response = result["response"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG error: {exc}") from exc

    return RagResponse(response=response)
