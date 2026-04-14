"""Tests for the Chainlit application entrypoint."""
import importlib
import asyncio
import json
import os
import sys
import types
import urllib.error
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _install_chainlit_stub() -> None:
    cl_stub = types.ModuleType("chainlit")

    def _decorator(func):
        return func

    class DummyMessage:
        def __init__(self, content, elements=None):
            self.content = content
            self.elements = elements or []

        async def send(self):
            await asyncio.sleep(0)
            return None

    class DummyStarter:
        def __init__(self, label, message, command=None, icon=None):
            self.label = label
            self.message = message
            self.command = command
            self.icon = icon

    class DummyUser:
        pass

    class DummyUserSession:
        def __init__(self):
            self._data = {}

        def set(self, key, value):
            self._data[key] = value

        def get(self, key):
            return self._data.get(key)

    cl_stub.on_app_startup = _decorator
    cl_stub.on_chat_start = _decorator
    cl_stub.on_message = _decorator
    cl_stub.set_starters = _decorator
    cl_stub.Message = DummyMessage
    cl_stub.Starter = DummyStarter
    cl_stub.User = DummyUser
    cl_stub.user_session = DummyUserSession()
    sys.modules["chainlit"] = cl_stub


_install_chainlit_stub()
app_chainlit = importlib.import_module("src.app_chainlit")


def _build_config(endpoint: str = "http://127.0.0.1:8000/rag", timeout_seconds: float = 30.0):
    return SimpleNamespace(
        frontend=SimpleNamespace(
            title="Book recommandor system",
            subtitle="Tell the system what kind of book you want, and it will search the catalog for a fitting recommendation.",
            starters=[
                SimpleNamespace(
                    label="I want an epic fantasy with political intrigue, magic, and a large cast.",
                    message="I want an epic fantasy with political intrigue, magic, and a large cast.",
                ),
                SimpleNamespace(
                    label="Suggest a cozy mystery with a charming setting and an amateur sleuth.",
                    message="Suggest a cozy mystery with a charming setting and an amateur sleuth.",
                ),
            ],
            rag_endpoint=endpoint,
            timeout_seconds=timeout_seconds,
        )
    )


class TestAppChainlit(unittest.TestCase):
    def setUp(self):
        app_chainlit._RAG_ENDPOINT = None
        app_chainlit._RAG_ERROR = None
        app_chainlit._APP_CONFIG = None

    def test_get_env_returns_value(self):
        """Test that environment variables are correclty loaded"""
        with patch.dict(os.environ, {"STRATEGY": "openai"}):
            value = app_chainlit.get_env(["STRATEGY"], None, True)
        self.assertEqual(value, "openai")

    def test_get_env_required_missing_raises(self):
        """Test that an error is raised if the variable is missing in the .env file"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                app_chainlit.get_env(["MISSING"], None, True)

    def test_on_app_startup_sets_globals(self):
        """Test globals setup"""
        with patch("src.app_chainlit.load_settings", return_value=_build_config()):
            app_chainlit.on_app_startup()

        self.assertEqual(app_chainlit._RAG_ENDPOINT, "http://127.0.0.1:8000/rag")
        self.assertIsNone(app_chainlit._RAG_ERROR)
        self.assertIsNotNone(app_chainlit._APP_CONFIG)

    def test_get_config_value_uses_get_method_when_available(self):
        """Test that _get_config_value first tries to use the get method of the config section, and falls back to attribute access if get is not available."""
        value = app_chainlit._get_config_value({"timeout_seconds": 12}, "timeout_seconds", "30")
        self.assertEqual(value, "12")

    def test_get_config_value_uses_attribute_access_as_fallback(self):
        """Test that _get_config_value falls back to attribute access if the get method is not available on the config section."""
        config_section = SimpleNamespace(timeout_seconds=15)
        value = app_chainlit._get_config_value(config_section, "timeout_seconds", "30")
        self.assertEqual(value, "15")

    def test_get_starter_specs_returns_empty_without_config(self):
        """Test that _get_starter_specs returns an empty list if there is no frontend config or if the starters key is missing from the frontend config."""
        self.assertEqual(app_chainlit._get_starter_specs(), [])

    def test_get_starter_specs_supports_mapping_frontend_and_skips_invalid_starters(self):
        """Test that _get_starter_specs can handle the frontend config being a mapping (e.g. dict) instead of an object with attributes, and that it skips starters that are missing a label or message."""
        app_chainlit._APP_CONFIG = SimpleNamespace(
            frontend={
                "starters": [
                    {"label": "Find fantasy", "message": "Find me a fantasy novel"},
                    {"label": " ", "message": "missing label"},
                    {"label": "missing message", "message": " "},
                ]
            }
        )

        starters = app_chainlit._get_starter_specs()

        self.assertEqual(
            starters,
            [{"label": "Find fantasy", "message": "Find me a fantasy novel"}],
        )

    def test_resolve_rag_endpoint_reads_settings(self):
        """Test that the RAG endpoint is correctly read from the settings during resolution."""
        with patch("src.app_chainlit.load_settings", return_value=_build_config(endpoint="http://api/rag")):
            endpoint = app_chainlit._resolve_rag_endpoint()

        self.assertEqual(endpoint, "http://api/rag")

    def test_sync_post_json_returns_http_error_payload(self):
        """Test that _sync_post_json correctly returns the payload of an HTTPError raised during the request, and that it returns the correct status code and error message."""
        error = urllib.error.HTTPError(
            url="http://127.0.0.1:8000/rag",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=MagicMock(read=MagicMock(return_value=b'{"detail":"backend unavailable"}')),
        )

        with patch("src.app_chainlit.urllib.request.urlopen", side_effect=error):
            status, body = app_chainlit._sync_post_json(
                "http://127.0.0.1:8000/rag",
                {"query": "hello"},
                5.0,
            )

        self.assertEqual(status, 503)
        self.assertEqual(body, '{"detail":"backend unavailable"}')

    def test_sync_post_json_raises_clear_error_for_unreachable_backend(self):
        """Test that _sync_post_json raises a readable RuntimeError when the backend is unreachable."""
        error = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

        with patch("src.app_chainlit.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Could not reach RAG endpoint"):
                app_chainlit._sync_post_json(
                    "http://127.0.0.1:8000/rag",
                    {"query": "hello", "thread_id": "thread-1"},
                    5.0,
                )

    def test_on_app_startup_captures_startup_errors(self):
        """Test that startup errors are captured and stored in _RAG_ERROR."""
        with patch("src.app_chainlit.load_settings", side_effect=RuntimeError("config boom")):
            app_chainlit.on_app_startup()

        self.assertIsNone(app_chainlit._APP_CONFIG)
        self.assertIsNone(app_chainlit._RAG_ENDPOINT)
        self.assertEqual(str(app_chainlit._RAG_ERROR), "config boom")


class TestAppChainlitAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app_chainlit._RAG_ENDPOINT = None
        app_chainlit._RAG_ERROR = None
        app_chainlit._APP_CONFIG = None

    async def test_on_chat_start_sets_session_without_sending_message(self):
        """Test that the Chainlit app prepares the session without collapsing the empty state."""
        app_chainlit._RAG_ENDPOINT = "http://127.0.0.1:8000/rag"
        app_chainlit._RAG_ERROR = None
        app_chainlit._APP_CONFIG = _build_config()

        with patch("src.app_chainlit.cl.user_session.set") as mock_set, patch(
            "src.app_chainlit.cl.Message"
        ) as mock_msg:
            await app_chainlit.on_chat_start()

        mock_set.assert_any_call("rag_endpoint", "http://127.0.0.1:8000/rag")
        mock_set.assert_any_call("rag_ready", True)
        mock_set.assert_any_call("thread_id", unittest.mock.ANY)
        mock_msg.assert_not_called()

    async def test_set_starters_returns_configured_starters(self):
        """Test that starter prompts are exposed from the frontend config."""
        app_chainlit._APP_CONFIG = _build_config()

        starters = await app_chainlit.set_starters()

        self.assertEqual(len(starters), 2)
        self.assertEqual(
            starters[0].label,
            "I want an epic fantasy with political intrigue, magic, and a large cast.",
        )
        self.assertEqual(
            starters[0].message,
            "I want an epic fantasy with political intrigue, magic, and a large cast.",
        )

    async def test_call_rag_endpoint_handles_invalid_json_body(self):
        """Test that a RuntimeError is raised when the RAG endpoint returns a non-JSON response, which is considered a malformed response."""
        with patch(
            "src.app_chainlit.asyncio.to_thread",
            new=AsyncMock(return_value=(200, "not json")),
        ):
            with self.assertRaisesRegex(RuntimeError, "malformed API response"):
                await app_chainlit._call_rag_endpoint("http://127.0.0.1:8000/rag", "hello", "thread-1", 30.0)

    async def test_call_rag_endpoint_raises_runtime_error_for_http_status(self):
        """Test that a RuntimeError is raised when the RAG endpoint returns a non-200 HTTP status."""
        with patch(
            "src.app_chainlit.asyncio.to_thread",
            new=AsyncMock(return_value=(500, json.dumps({"detail": "backend failed"}))),
        ):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                await app_chainlit._call_rag_endpoint("http://127.0.0.1:8000/rag", "hello", "thread-1", 30.0)

    async def test_call_rag_endpoint_replaces_invalid_recommendations_with_empty_list(self):
        """Test that invalid recommendations are replaced with an empty list."""
        with patch(
            "src.app_chainlit.asyncio.to_thread",
            new=AsyncMock(return_value=(200, json.dumps({"response": "ok", "recommendations": "oops", "sources": "oops"}))),
        ):
            payload = await app_chainlit._call_rag_endpoint("http://127.0.0.1:8000/rag", "hello", "thread-1", 30.0)

        self.assertEqual(payload, {"response": "ok", "recommendations": [], "sources": []})

    async def test_on_message_sends_response(self):
        """Test that the application answer user's message"""
        incoming = MagicMock()
        incoming.content = "hello"
        app_chainlit._APP_CONFIG = _build_config()

        card_instance = MagicMock()
        card_instance.send = AsyncMock()

        with patch(
            "src.app_chainlit.cl.user_session.get",
            side_effect=[True, "http://127.0.0.1:8000/rag", "thread-1"],
        ), patch(
            "src.app_chainlit._call_rag_endpoint",
            return_value={
                "response": "ok",
                "recommendations": [
                    {
                        "title": "Book A",
                        "author": "Alice",
                        "summary": "Summary",
                        "thumbnail": "https://example.com/a.jpg",
                        "num_pages": 320,
                    }
                ],
                "sources": [],
            },
        ), patch(
            "src.app_chainlit.cl.Message",
            return_value=card_instance,
        ) as mock_message:
            await app_chainlit.on_message(incoming)

        card_instance.send.assert_awaited_once()
        self.assertEqual(mock_message.call_count, 1)

    async def test_on_message_handles_error(self):
        """Test correct error raised when there is an error while receiving a message"""
        message_instance = MagicMock()
        message_instance.send = AsyncMock()
        incoming = MagicMock()
        incoming.content = "hello"
        app_chainlit._APP_CONFIG = _build_config()

        with patch(
            "src.app_chainlit.cl.user_session.get",
            side_effect=[True, "http://127.0.0.1:8000/rag", "thread-1"],
        ), patch(
            "src.app_chainlit._call_rag_endpoint",
            side_effect=RuntimeError("boom"),
        ), patch("src.app_chainlit.cl.Message", return_value=message_instance) as mock_message:
            await app_chainlit.on_message(incoming)

        mock_message.assert_called_once_with(content="RAG error: boom")
        message_instance.send.assert_awaited_once()

    async def test_on_chat_start_sends_startup_error_when_backend_failed(self):
        """Test that a message is sent to the user when the RAG endpoint failed to start up correctly, and that the error details are included in the message."""
        message_instance = MagicMock()
        message_instance.send = AsyncMock()
        app_chainlit._RAG_ERROR = RuntimeError("startup boom")

        with patch("src.app_chainlit.cl.Message", return_value=message_instance) as mock_message:
            await app_chainlit.on_chat_start()

        mock_message.assert_called_once_with(content="RAG startup error: startup boom")
        message_instance.send.assert_awaited_once()

    async def test_on_chat_start_sends_message_when_endpoint_is_missing(self):
        """Test that a message is sent to the user when the RAG endpoint is not configured."""
        message_instance = MagicMock()
        message_instance.send = AsyncMock()

        with patch("src.app_chainlit.cl.Message", return_value=message_instance) as mock_message:
            await app_chainlit.on_chat_start()

        mock_message.assert_called_once_with(content="RAG endpoint is not configured yet.")
        message_instance.send.assert_awaited_once()

    async def test_on_message_rejects_requests_when_session_is_not_ready(self):
        """Test that a message is sent to the user when they send a message before the session is ready."""
        message_instance = MagicMock()
        message_instance.send = AsyncMock()
        incoming = MagicMock()
        incoming.content = "hello"

        with patch("src.app_chainlit.cl.user_session.get", return_value=False), patch(
            "src.app_chainlit.cl.Message",
            return_value=message_instance,
        ) as mock_message:
            await app_chainlit.on_message(incoming)

        mock_message.assert_called_once_with(
            content="Received message while Chainlit session is not ready. Please wait a moment and try again."
        )
        message_instance.send.assert_awaited_once()

    async def test_on_message_sends_plain_response_without_recommendations(self):
        """Test that a plain response is sent when there are no recommendations."""
        message_instance = MagicMock()
        message_instance.send = AsyncMock()
        incoming = MagicMock()
        incoming.content = "hello"
        app_chainlit._APP_CONFIG = _build_config()

        with patch(
            "src.app_chainlit.cl.user_session.get",
            side_effect=[True, "http://127.0.0.1:8000/rag", "thread-1"],
        ), patch(
            "src.app_chainlit._call_rag_endpoint",
            return_value={"response": "plain response", "recommendations": [], "sources": []},
        ), patch("src.app_chainlit.cl.Message", return_value=message_instance) as mock_message:
            await app_chainlit.on_message(incoming)

        mock_message.assert_called_once_with(content="plain response")
        message_instance.send.assert_awaited_once()

    async def test_on_message_appends_sources_to_plain_response(self):
        """Test that follow-up answers include a rendered sources section."""
        message_instance = MagicMock()
        message_instance.send = AsyncMock()
        incoming = MagicMock()
        incoming.content = "hello"
        app_chainlit._APP_CONFIG = _build_config()

        with patch(
            "src.app_chainlit.cl.user_session.get",
            side_effect=[True, "http://127.0.0.1:8000/rag", "thread-1"],
        ), patch(
            "src.app_chainlit._call_rag_endpoint",
            return_value={
                "response": "plain response",
                "recommendations": [],
                "sources": [{"title": "Source A", "url": "https://example.com"}],
            },
        ), patch("src.app_chainlit.cl.Message", return_value=message_instance) as mock_message:
            await app_chainlit.on_message(incoming)

        mock_message.assert_called_once()
        sent_content = mock_message.call_args.kwargs["content"]
        self.assertIn("plain response", sent_content)
        self.assertIn("Sources", sent_content)


if __name__ == "__main__":
    unittest.main()
