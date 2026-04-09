"""Tests for the FastAPI application entrypoint."""
import importlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from pydantic import SecretStr

app_fastapi = importlib.import_module("src.app_fastapi")


def _build_config(
    *,
    strategy: str = "openai",
    embeddings_model: str = "embedding-model",
    embeddings_base_url=None,
    from_scratch: bool = False,
    data_path: str = "data/books.csv",
    columns_to_drop=None,
    index_path: str = "/tmp/index",
    index_name: str = "books_index",
    llm_provider: str = "openai",
    llm_model: str = "gpt-3.5-turbo",
    llm_base_url=None,
    retriever_k: int = 5,
):
    return SimpleNamespace(
        embeddings=SimpleNamespace(
            strategy=strategy,
            model=embeddings_model,
            base_url=embeddings_base_url,
        ),
        index=SimpleNamespace(
            from_scratch=from_scratch,
            data_path=data_path,
            columns_to_drop=columns_to_drop or ["isbn13", "isbn10"],
            path=index_path,
            name=index_name,
        ),
        llm=SimpleNamespace(
            provider=llm_provider,
            model=llm_model,
            base_url=llm_base_url,
        ),
        rag=SimpleNamespace(
            retriever_k=retriever_k,
        ),
    )


class TestAppFastapi(unittest.TestCase):
    def setUp(self):
        """Reset module-level globals before each test."""
        app_fastapi._RAG_INSTANCE = None
        app_fastapi._RAG_ERROR = None
        app_fastapi._APP_CONFIG = None

    def test_get_env_returns_value(self):
        """Test that environment variables are correctly loaded."""
        with patch.dict(os.environ, {"STRATEGY": "openai"}):
            value = app_fastapi.get_env(["STRATEGY"], None, True)
        self.assertEqual(value, "openai")

    def test_get_env_required_missing_raises(self):
        """Test that an error is raised if the variable is missing in the .env file."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                app_fastapi.get_env(["MISSING"], None, True)

    def test_parse_bool_handles_none_truthy_falsey_and_invalid_values(self):
        """Test that _parse_bool correctly interprets various boolean representations."""
        self.assertTrue(app_fastapi._parse_bool(None, default=True))
        self.assertTrue(app_fastapi._parse_bool(" yes "))
        self.assertFalse(app_fastapi._parse_bool("off"))
        with self.assertRaises(ValueError):
            app_fastapi._parse_bool("maybe")

    def test_parse_csv_list_handles_none_and_filters_empty_items(self):
        """Test that _parse_csv_list correctly handles None values and filters out empty items."""
        self.assertEqual(app_fastapi._parse_csv_list(None), [])
        self.assertEqual(app_fastapi._parse_csv_list(None, default=["a"]), ["a"])
        self.assertEqual(
            app_fastapi._parse_csv_list(" title, , author ,, summary "),
            ["title", "author", "summary"],
        )

    @patch("src.app_fastapi.RAGService")
    @patch("src.app_fastapi.ChatOpenAI")
    @patch("src.app_fastapi.load_store")
    @patch("src.app_fastapi.get_embeddings")
    @patch("src.app_fastapi.load_settings")
    def test_init_rag_builds_dependencies(
        self,
        mock_load_settings,
        mock_get_embeddings,
        mock_load_store,
        mock_chat_openai,
        mock_rag_service,
    ):
        """Test that all the components of the RAG service are correctly built."""
        mock_embeddings = MagicMock(name="Embeddings")
        mock_vectorstore = MagicMock(name="Vectorstore")
        mock_llm = MagicMock(name="LLM")
        mock_rag = MagicMock(name="RAGService")

        mock_get_embeddings.return_value = mock_embeddings
        mock_load_store.return_value = mock_vectorstore
        mock_chat_openai.return_value = mock_llm
        mock_rag_service.return_value = mock_rag
        mock_load_settings.return_value = _build_config()

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "fallback-key",
                "LLM_API_KEY": "llm-key",
                "EMBEDDINGS_API_KEY": "emb-key",
            },
            clear=True,
        ):
            result = app_fastapi._init_rag()

        self.assertIs(result, mock_rag)
        mock_get_embeddings.assert_called_once()
        embeddings_call_kwargs = mock_get_embeddings.call_args.kwargs
        self.assertEqual(embeddings_call_kwargs["strategy"], "openai")
        self.assertEqual(embeddings_call_kwargs["model"], "embedding-model")
        self.assertIsInstance(embeddings_call_kwargs["api_key"], SecretStr)
        self.assertEqual(embeddings_call_kwargs["api_key"].get_secret_value(), "emb-key")
        self.assertIsNone(embeddings_call_kwargs["base_url"])
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
        self.assertEqual(call_kwargs["api_key"].get_secret_value(), "llm-key")
        mock_rag_service.assert_called_once_with(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            k=5,
        )

    @patch("src.app_fastapi.RAGService")
    @patch("src.app_fastapi.ChatOpenAI")
    @patch("src.app_fastapi.create_database")
    @patch("src.app_fastapi.get_embeddings")
    @patch("src.app_fastapi.load_settings")
    def test_init_rag_builds_vectorstore_from_scratch(
        self,
        mock_load_settings,
        mock_get_embeddings,
        mock_create_database,
        mock_chat_openai,
        mock_rag_service,
    ):
        """Test that the vectorstore is built from scratch when the config specifies it, and that the correct parameters are passed to create_database."""
        mock_embeddings = MagicMock(name="Embeddings")
        mock_vectorstore = MagicMock(name="Vectorstore")
        mock_llm = MagicMock(name="LLM")
        mock_rag = MagicMock(name="RAGService")

        mock_load_settings.return_value = _build_config(from_scratch=True)
        mock_get_embeddings.return_value = mock_embeddings
        mock_create_database.return_value = mock_vectorstore
        mock_chat_openai.return_value = mock_llm
        mock_rag_service.return_value = mock_rag

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "fallback-key",
                "LLM_API_KEY": "llm-key",
                "EMBEDDINGS_API_KEY": "emb-key",
            },
            clear=True,
        ):
            result = app_fastapi._init_rag()

        self.assertIs(result, mock_rag)
        mock_create_database.assert_called_once_with(
            data_path="data/books.csv",
            columns_to_drop=["isbn13", "isbn10"],
            embeddings=mock_embeddings,
            index_path="/tmp/index",
            index_name="books_index",
        )

    @patch("src.app_fastapi.load_settings")
    def test_init_rag_rejects_unsupported_llm_provider(self, mock_load_settings):
        """Test that _init_rag raises an error if the LLM provider specified in the config is not supported."""
        mock_load_settings.return_value = _build_config(llm_provider="anthropic")

        with patch.dict(
            os.environ,
            {"EMBEDDINGS_API_KEY": "emb-key"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Only 'openai' is currently supported"):
                app_fastapi._init_rag()

    @patch("src.app_fastapi._init_rag")
    @patch("src.app_fastapi.load_settings")
    def test_on_app_startup_sets_globals(self, mock_load_settings, mock_init_rag):
        """Test globals setup."""
        mock_rag = MagicMock(name="RAGService")
        mock_init_rag.return_value = mock_rag
        mock_load_settings.return_value = _build_config()
        app_fastapi._APP_CONFIG = None

        app_fastapi._RAG_INSTANCE = None
        app_fastapi._RAG_ERROR = None
        app_fastapi.on_app_startup()

        self.assertIs(app_fastapi._RAG_INSTANCE, mock_rag)
        self.assertIsNone(app_fastapi._RAG_ERROR)
        self.assertIsNotNone(app_fastapi._APP_CONFIG)

    @patch("src.app_fastapi._init_rag")
    @patch("src.app_fastapi.load_settings")
    def test_on_app_startup_captures_startup_errors(self, mock_load_settings, mock_init_rag):
        """Test that errors during startup are captured and stored in _RAG_ERROR, and that _RAG_INSTANCE remains None."""
        mock_load_settings.side_effect = RuntimeError("settings boom")
        mock_init_rag.return_value = MagicMock(name="RAGService")

        app_fastapi.on_app_startup()

        self.assertIsNone(app_fastapi._APP_CONFIG)
        self.assertIsNone(app_fastapi._RAG_INSTANCE)
        self.assertEqual(str(app_fastapi._RAG_ERROR), "settings boom")


class TestAppFastapiAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app_fastapi._RAG_INSTANCE = None
        app_fastapi._RAG_ERROR = None
        app_fastapi._APP_CONFIG = None

    async def test_rag_endpoint_returns_response(self):
        """Test that the endpoint returns the RAG response."""
        rag = MagicMock(name="RAGService")
        rag.answer_query.return_value = {
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
        }

        app_fastapi._RAG_INSTANCE = rag
        app_fastapi._RAG_ERROR = None

        payload = app_fastapi.RagRequest(query="hello")
        result = await app_fastapi.rag_endpoint(payload)

        self.assertEqual(result.response, "ok")
        self.assertEqual(len(result.recommendations), 1)

    async def test_rag_endpoint_handles_error(self):
        """Test that the endpoint maps RAG errors to HTTP 500."""
        rag = MagicMock(name="RAGService")
        rag.answer_query.side_effect = RuntimeError("boom")

        app_fastapi._RAG_INSTANCE = rag
        app_fastapi._RAG_ERROR = None

        payload = app_fastapi.RagRequest(query="hello")
        with self.assertRaises(Exception) as ctx:
            await app_fastapi.rag_endpoint(payload)

        self.assertIn("Failed to answer /rag request", str(ctx.exception))

    async def test_rag_endpoint_requires_ready(self):
        """Test that the endpoint is blocked until RAG is ready."""
        app_fastapi._RAG_INSTANCE = None
        app_fastapi._RAG_ERROR = None

        payload = app_fastapi.RagRequest(query="hello")
        with self.assertRaises(Exception) as ctx:
            await app_fastapi.rag_endpoint(payload)

        self.assertIn("RAG service is not ready yet", str(ctx.exception))

    async def test_rag_endpoint_reports_startup_error(self):
        """Test that the endpoint surfaces startup errors."""
        app_fastapi._RAG_INSTANCE = None
        app_fastapi._RAG_ERROR = RuntimeError("startup boom")

        payload = app_fastapi.RagRequest(query="hello")
        with self.assertRaises(Exception) as ctx:
            await app_fastapi.rag_endpoint(payload)

        self.assertIn("startup errored", str(ctx.exception))

    async def test_health_endpoint_returns_ok_when_ready(self):
        """Test that the health endpoint returns 'ok' when the RAG service is ready."""
        app_fastapi._RAG_INSTANCE = MagicMock(name="RAGService")
        app_fastapi._RAG_ERROR = None

        result = await app_fastapi.health_endpoint()

        self.assertEqual(result.status, "ok")

    async def test_health_endpoint_reports_startup_error(self):
        """Test that the health endpoint surfaces startup errors."""
        app_fastapi._RAG_INSTANCE = None
        app_fastapi._RAG_ERROR = RuntimeError("startup boom")

        with self.assertRaises(Exception) as ctx:
            await app_fastapi.health_endpoint()

        self.assertIn("Health check failed because startup errored", str(ctx.exception))

    async def test_health_endpoint_reports_service_not_ready(self):
        """Test that the health endpoint reports that the service is not ready if RAG is not initialized and there are no startup errors."""
        app_fastapi._RAG_INSTANCE = None
        app_fastapi._RAG_ERROR = None

        with self.assertRaises(Exception) as ctx:
            await app_fastapi.health_endpoint()

        self.assertIn("RAG service is not ready yet", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
