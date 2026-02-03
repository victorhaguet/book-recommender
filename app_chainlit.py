"""
This is the main Chainlit application file.
It proxies user queries to the FastAPI RAG endpoint and handles chat events.
Environment variables are loaded from a .env file.
Please refer to the README for setup instructions.

Command to run the Chainlit app:
chainlit run app_chainlit.py --port 8080
"""

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Optional, overload, Literal

from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

_RAG_ENDPOINT: Optional[str] = None
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


def _resolve_rag_endpoint() -> str:
    """
    Resolve the RAG FastAPI endpoint from environment variables.
    
    :return: The RAG endpoint URL
    :rtype: str
    """
    endpoint = _get_env(
        ["RAG_ENDPOINT", "RAG_API_URL", "FASTAPI_URL"],
        "http://127.0.0.1:8000/rag",
        True,
    )
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
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.getcode(), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


async def _call_rag_endpoint(endpoint: str, query: str, timeout_seconds: float) -> str:
    """
    Call the RAG FastAPI endpoint asynchronously.
    
    :param endpoint: The RAG endpoint URL
    :type endpoint: str
    :param query: The query string to send to the RAG endpoint
    :type query: str
    :param timeout_seconds: The timeout for the request in seconds
    :type timeout_seconds: float
    :return: The response from the RAG endpoint
    :rtype: str
    """
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
        raise RuntimeError(f"API error ({status}): {detail}")

    response = payload.get("response")
    if not response:
        raise RuntimeError("Malformed response from API")
    return response


@cl.on_app_startup
def on_app_startup() -> None:
    """
    Resolve the FastAPI endpoint once at startup.
    """
    global _RAG_ENDPOINT, _RAG_ERROR
    try:
        _RAG_ENDPOINT = _resolve_rag_endpoint()
        _RAG_ERROR = None
    except Exception as exc:
        _RAG_ENDPOINT = None
        _RAG_ERROR = exc


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("rag_ready", False)

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

    endpoint: str = cl.user_session.get("rag_endpoint")
    timeout_seconds = float(_get_env(["RAG_TIMEOUT_SECONDS", "TIMEOUT_SECONDS"], "30"))
    try:
        response = await _call_rag_endpoint(endpoint, message.content, timeout_seconds)
    except Exception as exc:
        response = f"RAG error: {exc}"
    await cl.Message(content=response).send()
