""" Test for the function get_embeddings"""
import unittest
from unittest.mock import MagicMock, patch
from pydantic import SecretStr

from src.indexing.embeddings import get_embeddings


MODEL_ERROR = "Model variable must be a non-empty string"
API_KEY_ERROR = "Missing API key for OpenAI embeddings strategy"

class TestGetEmbeddings(unittest.TestCase):
    # -------------------------
    # Model validation
    # -------------------------
    def test_get_embeddings_model_none_raises(self):
        """Test scenario where the model isn't specified"""
        with self.assertRaises(ValueError) as ctx:
            get_embeddings(strategy="openai", model=None, api_key=SecretStr("sk-test"))  # type: ignore[arg-type]
        self.assertIn(MODEL_ERROR, str(ctx.exception))

    def test_get_embeddings_model_empty_raises(self):
        """ Test if model is an empty str"""
        with self.assertRaises(ValueError) as ctx:
            get_embeddings(strategy="openai", model="", api_key=SecretStr("sk-test"))
        self.assertIn(MODEL_ERROR, str(ctx.exception))

    def test_get_embeddings_model_not_string_raises(self):
        """ Test if model is not a str """
        with self.assertRaises(ValueError) as ctx:
            get_embeddings(strategy="openai", model=123, api_key=SecretStr("sk-test"))  # type: ignore[arg-type]
        self.assertIn(MODEL_ERROR, str(ctx.exception))

    # -------------------------
    # Strategy normalization
    # -------------------------
    @patch("src.indexing.embeddings.OpenAIEmbeddings")
    def test_get_embeddings_strategy_normalized_openai(self, mock_openai_cls):
        """ Test that strategy normalisation is working properly"""
        fake_instance = MagicMock(name="OpenAIEmbeddingsInstance")
        mock_openai_cls.return_value = fake_instance

        emb = get_embeddings(
            strategy="  OPENAI  ",
            model="text-embedding-3-small",
            api_key=SecretStr("sk-test"),
            base_url=None,
        )

        self.assertIs(emb, fake_instance)
        mock_openai_cls.assert_called_once()

    # -------------------------
    # OpenAI branch
    # -------------------------
    def test_get_embeddings_openai_missing_api_key_raises(self):
        """Test if api_key is none"""
        with self.assertRaises(ValueError) as ctx:
            get_embeddings(strategy="openai", model="text-embedding-3-small", api_key=None)
        self.assertIn(API_KEY_ERROR, str(ctx.exception))

    def test_get_embeddings_openai_empty_api_key_raises(self):
        """ Test if api_key is an empty str """
        with self.assertRaises(ValueError) as ctx:
            get_embeddings(strategy="openai", model="text-embedding-3-small", api_key=SecretStr(""))
        self.assertIn(API_KEY_ERROR, str(ctx.exception))

    def test_get_embeddings_openai_non_string_api_key_raises(self):
        """Test if api_key is an int variable"""
        with self.assertRaises(ValueError) as ctx:
            get_embeddings(strategy="openai", model="text-embedding-3-small", api_key=123)  # type: ignore[arg-type]
        self.assertIn(API_KEY_ERROR, str(ctx.exception))

    @patch("src.indexing.embeddings.OpenAIEmbeddings")
    def test_get_embeddings_openai_without_base_url(self, mock_openai_cls):
        """Test happy path call for OPENAI URL"""
        fake_instance = MagicMock(name="OpenAIEmbeddingsInstance")
        mock_openai_cls.return_value = fake_instance

        emb = get_embeddings(
            strategy="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("sk-test"),
            base_url=None,
        )

        self.assertIs(emb, fake_instance)
        mock_openai_cls.assert_called_once()
        _, kwargs = mock_openai_cls.call_args
        self.assertEqual(kwargs["model"], "text-embedding-3-small")
        self.assertEqual(kwargs["api_key"].get_secret_value(), "sk-test")
        self.assertIsNone(kwargs["base_url"])

    @patch("src.indexing.embeddings.OpenAIEmbeddings")
    def test_get_embeddings_openai_with_base_url(self, mock_openai_cls):
        """Test happy path with OpenAI-compatible API URL"""
        fake_instance = MagicMock(name="OpenAIEmbeddingsInstance")
        mock_openai_cls.return_value = fake_instance

        emb = get_embeddings(
            strategy="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("sk-test"),
            base_url="http://localhost:8000/v1",
        )

        self.assertIs(emb, fake_instance)
        mock_openai_cls.assert_called_once()
        _, kwargs = mock_openai_cls.call_args
        self.assertEqual(kwargs["model"], "text-embedding-3-small")
        self.assertEqual(kwargs["api_key"].get_secret_value(), "sk-test")
        self.assertEqual(kwargs["base_url"], "http://localhost:8000/v1")

    @patch("src.indexing.embeddings.OpenAIEmbeddings")
    def test_get_embeddings_openai_base_url_empty_string_not_added(self, mock_openai_cls):
        """Test empty base_url str value"""
        fake_instance = MagicMock(name="OpenAIEmbeddingsInstance")
        mock_openai_cls.return_value = fake_instance

        emb = get_embeddings(
            strategy="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("sk-test"),
            base_url="",
        )

        self.assertIs(emb, fake_instance)
        mock_openai_cls.assert_called_once()
        _, kwargs = mock_openai_cls.call_args
        self.assertEqual(kwargs["model"], "text-embedding-3-small")
        self.assertEqual(kwargs["api_key"].get_secret_value(), "sk-test")
        self.assertEqual(kwargs["base_url"], "")

    # -------------------------
    # HF branch
    # -------------------------
    @patch("src.indexing.embeddings.HuggingFaceEmbeddings")
    def test_get_embeddings_hf(self, mock_hf_cls):
        """Test HF happy path"""
        fake_instance = MagicMock(name="HuggingFaceEmbeddingsInstance")
        mock_hf_cls.return_value = fake_instance

        emb = get_embeddings(strategy="hf", model="sentence-transformers/all-MiniLM-L6-v2")

        self.assertIs(emb, fake_instance)
        mock_hf_cls.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # -------------------------
    # Invalid strategy
    # -------------------------
    def test_get_embeddings_invalid_strategy_raises(self):
        """Test invalid strategy scenario"""
        with self.assertRaises(ValueError) as ctx:
            get_embeddings(strategy="bad", model="some-model", api_key=SecretStr("sk-test"))
        self.assertIn("Unsupported embeddings strategy:", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
