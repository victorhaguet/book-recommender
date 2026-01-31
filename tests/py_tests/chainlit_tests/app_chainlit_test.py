"""Tests for the Chainlit application entrypoint."""
import importlib
import asyncio
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _install_chainlit_stub() -> None:
    cl_stub = types.ModuleType("chainlit")

    def _decorator(func):
        return func

    class DummyMessage:
        def __init__(self, content):
            self.content = content

        async def send(self):
            await asyncio.sleep(0)
            return None

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
    cl_stub.Message = DummyMessage
    cl_stub.user_session = DummyUserSession()
    sys.modules["chainlit"] = cl_stub


_install_chainlit_stub()
app_chainlit = importlib.import_module("app_chainlit")


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

    @patch("app_chainlit.RAGService")
    @patch("app_chainlit.ChatOpenAI")
    @patch("app_chainlit.load_store")
    @patch("app_chainlit.get_embeddings")
    def test_init_rag_builds_dependencies(
        self,
        mock_get_embeddings,
        mock_load_store,
        mock_chat_openai,
        mock_rag_service,
    ):
        """Test that all the components of the RAG service are correctly build"""
        mock_embeddings = MagicMock(name="Embeddings")
        mock_vectorstore = MagicMock(name="Vectorstore")
        mock_llm = MagicMock(name="LLM")
        mock_rag = MagicMock(name="RAGService")

        mock_get_embeddings.return_value = mock_embeddings
        mock_load_store.return_value = mock_vectorstore
        mock_chat_openai.return_value = mock_llm
        mock_rag_service.return_value = mock_rag

        with patch.dict(
            os.environ,
            {
                "STRATEGY": "openai",
                "EMBEDDINGS_MODEL": "embedding-model",
                "OPENAI_API_KEY": "test-key",
                "FROM_SCRATCH": "false",
                "FAISS_INDEX_PATH": "/tmp/index",
                "FAISS_INDEX_NAME": "books_index",
            },
            clear=True,
        ):
            result = app_chainlit._init_rag()

        self.assertIs(result, mock_rag)
        mock_get_embeddings.assert_called_once_with(
            strategy="openai",
            model="embedding-model",
            api_key="test-key",
            base_url=None,
        )
        mock_load_store.assert_called_once_with(
            embeddings=mock_embeddings,
            path="/tmp/index",
            index_name="books_index",
        )
        self.assertTrue(mock_chat_openai.called)
        call_kwargs = mock_chat_openai.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-3.5-turbo")
        self.assertEqual(call_kwargs["temperature"], 0)
        self.assertIsNone(call_kwargs["base_url"])
        self.assertEqual(call_kwargs["api_key"].get_secret_value(), "test-key")
        mock_rag_service.assert_called_once_with(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            k=5,
        )

    @patch("app_chainlit._init_rag")
    def test_on_app_startup_sets_globals(self, mock_init_rag):
        """Test globals setup"""
        mock_rag = MagicMock(name="RAGService")
        mock_init_rag.return_value = mock_rag

        app_chainlit._RAG_INSTANCE = None
        app_chainlit._RAG_ERROR = None
        app_chainlit.on_app_startup()

        self.assertIs(app_chainlit._RAG_INSTANCE, mock_rag)
        self.assertIsNone(app_chainlit._RAG_ERROR)


class TestAppChainlitAsync(unittest.IsolatedAsyncioTestCase):
    async def test_on_chat_start_sets_session_and_sends_message(self):
        """Test that the chainlit app correctly start and send the introduction message"""
        rag = MagicMock(name="RAGService")
        message_instance = MagicMock()
        message_instance.send = AsyncMock()

        app_chainlit._RAG_INSTANCE = rag
        app_chainlit._RAG_ERROR = None

        with patch("app_chainlit.cl.user_session.set") as mock_set, patch(
            "app_chainlit.cl.Message", return_value=message_instance
        ) as mock_msg:
            await app_chainlit.on_chat_start()

        mock_set.assert_any_call("rag", rag)
        mock_set.assert_any_call("rag_ready", True)
        mock_msg.assert_called_once_with(content="Hi! What do you want to read?")
        message_instance.send.assert_awaited_once()

    async def test_on_message_sends_response(self):
        """Test that the application answer user's message"""
        rag = MagicMock(name="RAGService")
        rag.answer_query.return_value = {"response": "ok"}
        message_instance = MagicMock()
        message_instance.send = AsyncMock()
        incoming = MagicMock()
        incoming.content = "hello"

        with patch(
            "app_chainlit.cl.user_session.get", side_effect=[True, rag]
        ), patch("app_chainlit.cl.Message", return_value=message_instance) as mock_message:
            await app_chainlit.on_message(incoming)

        message_instance.send.assert_awaited_once()
        mock_message.assert_called_once_with(content="ok")

    async def test_on_message_handles_error(self):
        """Test correct error raised when there is an error while receiving a message"""

        rag = MagicMock(name="RAGService")
        rag.answer_query.side_effect = RuntimeError("boom")
        message_instance = MagicMock()
        message_instance.send = AsyncMock()
        incoming = MagicMock()
        incoming.content = "hello"

        with patch(
            "app_chainlit.cl.user_session.get", side_effect=[True, rag]
        ), patch("app_chainlit.cl.Message", return_value=message_instance)as mock_message:
            await app_chainlit.on_message(incoming)

        mock_message.assert_called_once_with(content="RAG error: boom")
        message_instance.send.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
