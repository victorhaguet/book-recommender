"""
This is the main Chainlit application file. 
It initializes the RAG service and handles chat events. 
Environment variables are loaded from a .env file.
Please refer to the README for setup instructions. 
"""

import os
from typing import Optional, overload, Literal, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from pydantic import SecretStr
import chainlit as cl
from src.indexing.embeddings import get_embeddings
from src.indexing.faiss_store import load_store
from src.indexing.create_database import create_database
from src.rag.rag_service import RAGService

load_dotenv()

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
    # Embedding model initialization
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

    # Get database
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
            index_name=index_name
        )
    else:
        vectorstore=load_store(embeddings=embeddings, path=index_path, index_name=index_name)


    chat_model = _get_env(["CHAT_MODEL", "LLM_MODEL"], "gpt-3.5-turbo", True)
    llm = ChatOpenAI(model=chat_model, api_key=SecretStr(api_key), base_url=base_url, temperature=0)
    k = int(_get_env(["RETRIEVER_K", "K"], "5"))

    return RAGService(llm=llm, vectorstore=vectorstore, k=k)


@cl.on_app_startup
def on_app_startup() -> None:
    """
    Build/load the vectorstore once at app startup so users cannot chat before it's ready.
    """
    global _RAG_INSTANCE, _RAG_ERROR
    try:
        _RAG_INSTANCE = _init_rag()
        _RAG_ERROR = None
    except Exception as exc:
        _RAG_INSTANCE = None
        _RAG_ERROR = exc


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("rag_ready", False)

    if _RAG_ERROR is not None:
        await cl.Message(
            content=f"RAG startup error: {_RAG_ERROR}"
        ).send()
        return

    if _RAG_INSTANCE is None:
        await cl.Message(
            content="We are currently learning the books, please wait."
        ).send()
        return

    cl.user_session.set("rag", _RAG_INSTANCE)
    cl.user_session.set("rag_ready", True)
    await cl.Message(
        content="Hi! What do you want to read?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Chat message event handler.
    
    :param message: Message sent by the user
    :type message: cl.Message
    """
    rag_ready = cl.user_session.get("rag_ready")
    if not rag_ready:
        await cl.Message(
            content="We are currently learning the books, please wait."
        ).send()
        return

    rag: RAGService = cl.user_session.get("rag")
    try:
        result = rag.answer_query(message.content)
        response = result["response"]
    except Exception as exc:
        response = f"RAG error: {exc}"
    await cl.Message(content=response).send()
