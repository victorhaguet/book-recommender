"""Tests for the Chainlit application entrypoint."""
import importlib
import asyncio
import os
import sys
import types
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
            app_chainlit._RAG_ENDPOINT = None
            app_chainlit._RAG_ERROR = None
            app_chainlit._APP_CONFIG = None
            app_chainlit.on_app_startup()

        self.assertEqual(app_chainlit._RAG_ENDPOINT, "http://127.0.0.1:8000/rag")
        self.assertIsNone(app_chainlit._RAG_ERROR)
        self.assertIsNotNone(app_chainlit._APP_CONFIG)


class TestAppChainlitAsync(unittest.IsolatedAsyncioTestCase):
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

    async def test_on_message_sends_response(self):
        """Test that the application answer user's message"""
        incoming = MagicMock()
        incoming.content = "hello"
        app_chainlit._APP_CONFIG = _build_config()

        card_instance = MagicMock()
        card_instance.send = AsyncMock()

        with patch(
            "src.app_chainlit.cl.user_session.get",
            side_effect=[True, "http://127.0.0.1:8000/rag"],
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
            side_effect=[True, "http://127.0.0.1:8000/rag"],
        ), patch(
            "src.app_chainlit._call_rag_endpoint",
            side_effect=RuntimeError("boom"),
        ), patch("src.app_chainlit.cl.Message", return_value=message_instance) as mock_message:
            await app_chainlit.on_message(incoming)

        mock_message.assert_called_once_with(content="RAG error: boom")
        message_instance.send.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
