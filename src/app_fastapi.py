"""
This module contain the FastAPI application entrypoint for the RAG service.
It mirrors the Chainlit setup: the vectorstore is built/loaded on startup
and the RAG endpoint is unavailable until initialization completes.

Start to run the Fast API app : 
python3 -m uvicorn src.app_fastapi:app --reload

Query the RAG service:
curl -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"<your-query>\"}"
"""

import os
from typing import Optional, overload, Literal, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from omegaconf import DictConfig

from src.app_config import load_settings
from src.indexing.embeddings import get_embeddings
from src.indexing.faiss_store import load_store
from src.indexing.create_database import create_database
from src.logging_utils import get_logger, summarize_text
from src.rag.agentic_service import AgenticRAGService

load_dotenv()

logger = get_logger(__name__)

app = FastAPI()

_RAG_INSTANCE: Optional[AgenticRAGService] = None
_RAG_ERROR: Optional[Exception] = None
_APP_CONFIG: Optional[DictConfig] = None


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
        raise ValueError(f"Missing required environment variable. Tried keys: {keys}")
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
    raise ValueError(f"Invalid boolean environment value: {value}")


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


def _init_rag() -> AgenticRAGService:
    """
    Initialize the RAG service from environment variables.

    :return: Initialized RAG service
    :rtype: RAGService
    """
    logger.info("Initializing FastAPI RAG dependencies from environment")
    config = load_settings()
    strategy: str = config.embeddings.strategy
    embeddings_model: str = config.embeddings.model

    # Load API keys
    embeddings_api_key: Optional[str] = _get_env(
        ["EMBEDDINGS_API_KEY", "OPENAI_EMBEDDINGS_API_KEY", "OPENAI_API_KEY", "api_key"],
        None,
        strategy == "openai",
    )
    embeddings_base_url: Optional[str] = config.embeddings.base_url
    embeddings_api_secret: Optional[SecretStr] = None
    if embeddings_api_key is not None:
        embeddings_api_secret = SecretStr(embeddings_api_key)

    llm_provider: str = config.llm.provider.lower().strip()
    if llm_provider != "openai":
        raise ValueError("Only 'openai' is currently supported for the LLM provider") # This will be extended
    llm_model: str = config.llm.model
    llm_api_key: Optional[str] = _get_env(
        ["LLM_API_KEY", "OPENAI_LLM_API_KEY", "OPENAI_API_KEY", "api_key"],
        None,
        True,
    )
    llm_base_url: Optional[str] = config.llm.base_url
    llm_api_secret: Optional[SecretStr] = None
    if llm_api_key is not None:
        llm_api_secret = SecretStr(llm_api_key)

    # Initialize embeddings
    embeddings: Embeddings = get_embeddings(
        strategy=strategy,
        model=embeddings_model,
        api_key=embeddings_api_secret,
        base_url=embeddings_base_url,
    )

    from_scratch: bool = bool(config.index.from_scratch)
    index_path: str = config.index.path
    index_name: str = config.index.name
    data_path: str = config.index.data_path
    columns_to_drop: List[str] = list(config.index.columns_to_drop)
    min_description_words: int = int(config.index.min_description_words)

    vectorstore: FAISS
    if from_scratch:
        logger.info("Building vectorstore from scratch")
        vectorstore = create_database(
            data_path=data_path,
            columns_to_drop=columns_to_drop,
            min_description_words=min_description_words,
            embeddings=embeddings,
            index_path=index_path,
            index_name=index_name,
        )
    else:
        logger.info("Loading existing vectorstore from disk")
        vectorstore = load_store(embeddings=embeddings, path=index_path, index_name=index_name)

    # Initialize LLM
    llm = ChatOpenAI(
        model=llm_model,
        api_key=llm_api_secret,
        base_url=llm_base_url,
        temperature=0,
    )
    k = int(config.rag.retriever_k)

    classifier_model: str = str(config.agentic.classifier_model or llm_model)
    classifier_llm = ChatOpenAI(
        model=classifier_model,
        api_key=llm_api_secret,
        base_url=llm_base_url,
        temperature=0,
    )
    tavily_max_results = int(config.agentic.tavily_max_results)
    wikipedia_top_k_results = int(config.agentic.wikipedia_top_k_results)

    logger.info(
        "FastAPI RAG dependencies initialized successfully with chat model '%s', classifier model '%s' and k=%d",
        llm_model,
        classifier_model,
        k,
    )
    return AgenticRAGService(
        llm=llm,
        classifier_llm=classifier_llm,
        vectorstore=vectorstore,
        k=k,
        tavily_max_results=tavily_max_results,
        wikipedia_top_k_results=wikipedia_top_k_results,
    )


@app.on_event("startup")
def on_app_startup() -> None:
    """
    Build/load the vectorstore once at app startup so the endpoint is blocked until ready.
    """
    global _RAG_INSTANCE, _RAG_ERROR, _APP_CONFIG
    logger.info("FastAPI startup: preparing RAG service")
    try:
        _APP_CONFIG = load_settings()
        _RAG_INSTANCE = _init_rag()
        _RAG_ERROR = None
        logger.info("FastAPI startup completed successfully")
    except Exception as exc:
        _APP_CONFIG = None
        _RAG_INSTANCE = None
        _RAG_ERROR = exc
        logger.exception("FastAPI startup failed")


class RagRequest(BaseModel):
    """Request model for the RAG endpoint."""
    query: str
    thread_id: str

class RecommendationCard(BaseModel):
    """Model for a recommended book card in the RAG response."""
    title: str
    thumbnail: str | None = None
    author: str
    summary: str
    num_pages: int | str | None = None


class SourceLink(BaseModel):
    """External source attached to a follow-up answer."""
    title: str
    url: str


class RagResponse(BaseModel):
    """Response model for the RAG endpoint."""
    response: str
    recommendations: list[RecommendationCard] = Field(default_factory=list)
    sources: list[SourceLink] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """
    Healthcheck endpoint for container readiness.
    """
    if _RAG_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"Health check failed because startup errored: {_RAG_ERROR}")
    if _RAG_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Health check failed because RAG service is not ready yet.")
    return HealthResponse(status="ok")


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
    logger.info(
        "Received /rag request for thread '%s' and query: '%s'",
        payload.thread_id,
        summarize_text(payload.query),
    )

    # Check if it runs properly
    if _RAG_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"Rejecting /rag request because startup errored: {_RAG_ERROR}")

    # Check if the vectorstore is ready
    if _RAG_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Rejecting /rag request because RAG service is not ready yet.")

    # Try to call the RAG instance
    try:
        result = _RAG_INSTANCE.answer_query(payload.query, payload.thread_id)
        response = result["response"]
        recommendations = result.get("recommendations", [])
        sources = result.get("sources", [])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer /rag request: {exc}") from exc

    logger.info("Successfully answered /rag request with response length %d", len(response))
    return RagResponse(response=response, recommendations=recommendations, sources=sources)
