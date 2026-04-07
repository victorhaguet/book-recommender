"""
This is the main Chainlit application file.
It proxies user queries to the FastAPI RAG endpoint and handles chat events.
Environment variables are loaded from a .env file.
Please refer to the README for setup instructions.

Command to run the Chainlit app:
chainlit run src/app_chainlit.py --port 8080
"""

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional, overload, Literal

from dotenv import load_dotenv
import chainlit as cl
from omegaconf import DictConfig

from src.app_config import load_settings
from src.logging_utils import get_logger, summarize_text

load_dotenv()

logger = get_logger(__name__)

_RAG_ENDPOINT: Optional[str] = None
_RAG_ERROR: Optional[Exception] = None
_APP_CONFIG: Optional[DictConfig] = None


def _get_config_value(config_section: object, key: str, default: str) -> str:
    """
    Read a string value from either an OmegaConf section or a simple object.

    :param config_section: Frontend config section
    :type config_section: object
    :param key: Config key to read
    :type key: str
    :param default: Fallback value
    :type default: str
    :return: Resolved string value
    :rtype: str
    """
    if hasattr(config_section, "get"):
        value = config_section.get(key, default)
    else:
        value = getattr(config_section, key, default)
    return str(value)


def _get_starter_specs() -> list[dict[str, str]]:
    """
    Return the configured starter prompts.

    :return: Starter prompt specifications
    :rtype: list[dict[str, str]]
    """
    if _APP_CONFIG is None:
        return []

    if hasattr(_APP_CONFIG.frontend, "get"):
        starters = _APP_CONFIG.frontend.get("starters", [])
    else:
        starters = getattr(_APP_CONFIG.frontend, "starters", [])
    starter_specs: list[dict[str, str]] = []
    for starter in starters:
        if hasattr(starter, "get"):
            label = str(starter.get("label", "")).strip()
            message = str(starter.get("message", "")).strip()
        else:
            label = str(getattr(starter, "label", "")).strip()
            message = str(getattr(starter, "message", "")).strip()
        if not label or not message:
            continue
        starter_specs.append({"label": label, "message": message})
    return starter_specs


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


def _resolve_rag_endpoint() -> str:
    """
    Resolve the RAG FastAPI endpoint from environment variables.
    
    :return: The RAG endpoint URL
    :rtype: str
    """
    config = load_settings()
    endpoint = config.frontend.rag_endpoint
    logger.info("Resolved Chainlit RAG endpoint to '%s'", endpoint)
    return endpoint


def _sync_post_json(url: str, payload: dict, timeout_seconds: float) -> tuple[int, str]:
    """
    Synchronous helper to post JSON data.
    
    :param url: The URL to send the POST request to
    :type url: str
    :param payload: The JSON payload to send in the request body
    :type payload: dict
    :param timeout_seconds: The timeout for the request in seconds
    :type timeout_seconds: float
    :return: A tuple containing the HTTP status code and the response body as a string
    :rtype: tuple[int, str]
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        logger.info("Posting query to RAG endpoint '%s' with timeout %.1fs", url, timeout_seconds)
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.getcode(), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        logger.warning("RAG endpoint returned HTTP %d", exc.code)
        return exc.code, exc.read().decode("utf-8")


async def _call_rag_endpoint(endpoint: str, query: str, timeout_seconds: float) -> dict[str, Any]:
    """
    Call the RAG FastAPI endpoint asynchronously.
    
    :param endpoint: The RAG endpoint URL
    :type endpoint: str
    :param query: The query string to send to the RAG endpoint
    :type query: str
    :param timeout_seconds: The timeout for the request in seconds
    :type timeout_seconds: float
    :return: The parsed response from the RAG endpoint
    :rtype: dict[str, Any]
    """
    logger.info("Forwarding Chainlit query: '%s'", summarize_text(query))
    status, body = await asyncio.to_thread(
        _sync_post_json,
        endpoint,
        {"query": query},
        timeout_seconds,
    )
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    if status >= 400:
        detail = payload.get("detail", body or f"HTTP {status}")
        raise RuntimeError(f"Chainlit received API error ({status}): {detail}")

    response = payload.get("response")
    if not response:
        raise RuntimeError("Chainlit received malformed API response")
    recommendations = payload.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []
    return {"response": response, "recommendations": recommendations}


def _format_recommendation_message(card: dict[str, Any]) -> str:
    """
    Build the message text for a recommendation card.

    :param card: Structured recommendation card.
    :type card: dict[str, Any]
    :return: Markdown content for the message.
    :rtype: str
    """
    # Get the book values
    title = str(card.get("title", "Unknown title")).strip()
    thumbnail = str(card.get("thumbnail", "")).strip()
    author = str(card.get("author", "Unknown author")).strip()
    num_pages = card.get("num_pages")
    summary = str(card.get("summary", "")).strip()

    # Build the message content with conditional formatting based on available values
    first_line = title
    if author:
        first_line = f"{title} - {author}"

    parts = [f"### {first_line}"]
    if thumbnail:
        parts.append(f"![Book cover for {title}]({thumbnail})")
    if num_pages not in (None, "", "N/A"):
        parts.append(f"**Pages:** {num_pages}")
    if summary:
        parts.append(summary)
    return "\n\n".join(parts)


def _format_recommendations_response(
    intro_message: str,
    recommendations: list[dict[str, Any]],
) -> str:
    """
    Build a single markdown response containing the intro and all recommendations.

    :param intro_message: Introductory sentence shown before the list.
    :type intro_message: str
    :param recommendations: Structured recommendation cards.
    :type recommendations: list[dict[str, Any]]
    :return: Markdown response content.
    :rtype: str
    """
    sections = [intro_message.strip()]
    sections.extend(_format_recommendation_message(card) for card in recommendations)
    return "\n\n---\n\n".join(section for section in sections if section.strip())


@cl.on_app_startup
def on_app_startup() -> None:
    """
    Resolve the FastAPI endpoint once at startup.
    """
    global _RAG_ENDPOINT, _RAG_ERROR, _APP_CONFIG
    logger.info("Chainlit startup: resolving backend endpoint")
    try:
        _APP_CONFIG = load_settings()
        _RAG_ENDPOINT = _APP_CONFIG.frontend.rag_endpoint
        _RAG_ERROR = None
        logger.info("Chainlit startup completed successfully")
    except Exception as exc:
        _APP_CONFIG = None
        _RAG_ENDPOINT = None
        _RAG_ERROR = exc
        logger.exception("Chainlit startup failed")


@cl.set_starters
async def set_starters(_user: Optional[cl.User] = None) -> list[cl.Starter]:
    """
    Expose starter prompts on the initial Chainlit screen.

    :return: Starter prompts for the frontend
    :rtype: list[cl.Starter]
    """
    return [
        cl.Starter(label=starter["label"], message=starter["message"])
        for starter in _get_starter_specs()
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Initialize the Chainlit session for a new chat. Check if the RAG endpoint is ready and store it in the session.
    """
    cl.user_session.set("rag_ready", False)
    logger.info("New Chainlit chat session started")

    if _RAG_ERROR is not None:
        await cl.Message(
            content=f"RAG startup error: {_RAG_ERROR}"
        ).send()
        return

    if _RAG_ENDPOINT is None:
        await cl.Message(
            content="RAG endpoint is not configured yet."
        ).send()
        return

    cl.user_session.set("rag_endpoint", _RAG_ENDPOINT)
    cl.user_session.set("rag_ready", True)
    logger.info("Chat session is ready to forward requests")


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Chat message event handler.
    
    :param message: Message sent by the user
    :type message: cl.Message
    """
    logger.info("Received Chainlit message: '%s'", summarize_text(message.content))
    rag_ready = cl.user_session.get("rag_ready")
    if not rag_ready:
        await cl.Message(
            content="Received message while Chainlit session is not ready. Please wait a moment and try again."
        ).send()
        return

    endpoint: str = cl.user_session.get("rag_endpoint")
    timeout_seconds = 30.0 if _APP_CONFIG is None else float(_APP_CONFIG.frontend.timeout_seconds)
    try:
        payload = await _call_rag_endpoint(endpoint, message.content, timeout_seconds)
    except Exception as exc:
        logger.exception("Chainlit failed to get a response from the RAG backend")
        response = f"RAG error: {exc}"
        await cl.Message(content=response).send()
        return
    else:
        logger.info("Sending Chainlit response back to the user")

    response = str(payload.get("response", "")).strip()
    recommendations = payload.get("recommendations", [])
    if recommendations:
        intro_message = response or "Here are the books that could correspond to your expectations."
        await cl.Message(
            content=_format_recommendations_response(intro_message, recommendations)
        ).send()
    elif response:
        await cl.Message(content=response).send()
