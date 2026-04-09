""" Tests for FAISS store helpers."""
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.indexing.faiss_store import add_documents, create_faiss_db, load_store, save_store


FOLDER_PATH="/tmp/faiss"

class TestFaissStore(unittest.TestCase):
    def test_create_faiss_db_requires_embeddings(self):
        """Creating a FAISS store requires an embeddings instance."""
        with self.assertRaises(ValueError):
            create_faiss_db(None)

    @patch("src.indexing.faiss_store.InMemoryDocstore")
    @patch("src.indexing.faiss_store.FAISS")
    @patch("src.indexing.faiss_store.faiss.IndexFlatL2")
    def test_create_faiss_db_builds_empty_store(
        self,
        mock_index_cls,
        mock_faiss_cls,
        mock_docstore_cls,
    ):
        """Create a FAISS store with expected components."""
        embeddings = MagicMock(name="Embeddings")
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        fake_index = MagicMock(name="IndexFlatL2")
        fake_docstore = MagicMock(name="InMemoryDocstore")
        fake_store = MagicMock(name="FAISSStore")

        mock_index_cls.return_value = fake_index
        mock_docstore_cls.return_value = fake_docstore
        mock_faiss_cls.return_value = fake_store

        store = create_faiss_db(embeddings)

        self.assertIs(store, fake_store)
        embeddings.embed_query.assert_called_once_with("Hello books lovers !")
        mock_index_cls.assert_called_once_with(3)
        mock_docstore_cls.assert_called_once_with()
        mock_faiss_cls.assert_called_once_with(
            embedding_function=embeddings,
            index=fake_index,
            docstore=fake_docstore,
            index_to_docstore_id={},
        )

    def test_create_faiss_db_empty_embedding_vector_raises(self):
        """Empty embedding vector should fail when creating FAISS index."""
        embeddings = MagicMock(name="Embeddings")
        embeddings.embed_query.return_value = []

        with self.assertRaises(ValueError):
            create_faiss_db(embeddings)

    def test_create_faiss_db_embed_query_error_propagates(self):
        """Embedding errors should propagate."""
        embeddings = MagicMock(name="Embeddings")
        embeddings.embed_query.side_effect = RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            create_faiss_db(embeddings)

    def test_add_documents_calls_vectorstore(self):
        """Add documents delegates to vectorstore."""
        vectorstore = MagicMock(name="FAISSStore")
        documents = [Document(page_content="Doc 1"), Document(page_content="Doc 2")]

        add_documents(documents, vectorstore)

        vectorstore.add_documents.assert_called_once_with(documents=documents)

    def test_add_documents_empty_list_still_calls_vectorstore(self):
        """Empty documents list should still call vectorstore."""
        vectorstore = MagicMock(name="FAISSStore")
        documents = []

        add_documents(documents, vectorstore)

        vectorstore.add_documents.assert_called_once_with(documents=documents)

    def test_add_documents_vectorstore_error_propagates(self):
        """Vectorstore errors should propagate."""
        vectorstore = MagicMock(name="FAISSStore")
        vectorstore.add_documents.side_effect = ValueError("bad docs")
        documents = [Document(page_content="Doc 1")]

        with self.assertRaises(ValueError):
            add_documents(documents, vectorstore)

    def test_add_documents_requires_vectorstore(self):
        """Adding documents requires a valid vectorstore."""
        with self.assertRaises(ValueError):
            add_documents([Document(page_content="Doc 1")], None)

    def test_save_store_calls_vectorstore(self):
        """Save store delegates to vectorstore."""
        vectorstore = MagicMock(name="FAISSStore")

        save_store(vectorstore, path=FOLDER_PATH, index_name="test-index")

        vectorstore.save_local.assert_called_once_with(
            folder_path=FOLDER_PATH,
            index_name="test-index",
        )

    def test_save_store_error_propagates(self):
        """Save errors should propagate."""
        vectorstore = MagicMock(name="FAISSStore")
        vectorstore.save_local.side_effect = OSError("nope")

        with self.assertRaises(OSError):
            save_store(vectorstore, path=FOLDER_PATH, index_name="test-index")

    def test_save_store_requires_vectorstore(self):
        """Saving a store requires a valid vectorstore."""
        with self.assertRaises(ValueError):
            save_store(None, path=FOLDER_PATH, index_name="test-index")

    def test_save_store_requires_valid_path(self):
        """Saving a store requires a valid path."""
        vectorstore = MagicMock(name="FAISSStore")

        with self.assertRaises(ValueError):
            save_store(vectorstore, path="", index_name="test-index")

    def test_save_store_requires_valid_index_name(self):
        """Saving a store requires a valid index name."""
        vectorstore = MagicMock(name="FAISSStore")

        with self.assertRaises(ValueError):
            save_store(vectorstore, path=FOLDER_PATH, index_name="")

    @patch("src.indexing.faiss_store.FAISS.load_local")
    def test_load_store_returns_vectorstore(self, mock_load_local):
        """Load store returns FAISS vectorstore."""
        embeddings = MagicMock(name="Embeddings")
        fake_store = MagicMock(name="FAISSStore")
        mock_load_local.return_value = fake_store

        store = load_store(embeddings, path=FOLDER_PATH, index_name="test-index")

        self.assertIs(store, fake_store)
        mock_load_local.assert_called_once_with(
            folder_path=FOLDER_PATH,
            embeddings=embeddings,
            index_name="test-index",
            allow_dangerous_deserialization=True
        )

    @patch("src.indexing.faiss_store.FAISS.load_local")
    def test_load_store_error_propagates(self, mock_load_local):
        """Load errors should propagate."""
        embeddings = MagicMock(name="Embeddings")
        mock_load_local.side_effect = FileNotFoundError("missing")

        with self.assertRaises(FileNotFoundError):
            load_store(embeddings, path=FOLDER_PATH, index_name="test-index")

    def test_load_store_requires_embeddings(self):
        """Loading a store requires a valid embeddings instance."""
        with self.assertRaises(ValueError):
            load_store(None, path=FOLDER_PATH, index_name="test-index")

    def test_load_store_requires_valid_path(self):
        """Loading a store requires a valid path."""
        embeddings = MagicMock(name="Embeddings")

        with self.assertRaises(ValueError):
            load_store(embeddings, path="", index_name="test-index")

    def test_load_store_requires_valid_index_name(self):
        """Loading a store requires a valid index name."""
        embeddings = MagicMock(name="Embeddings")

        with self.assertRaises(ValueError):
            load_store(embeddings, path=FOLDER_PATH, index_name="")

    @patch("src.indexing.faiss_store.FAISS.load_local")
    def test_load_store_wraps_unexpected_errors(self, mock_load_local):
        """Unexpected errors during load should be wrapped in a RuntimeError with a clear message."""
        embeddings = MagicMock(name="Embeddings")
        mock_load_local.side_effect = RuntimeError("bad index")

        with self.assertRaisesRegex(RuntimeError, "Failed to load FAISS store"):
            load_store(embeddings, path=FOLDER_PATH, index_name="test-index")


if __name__ == "__main__":
    unittest.main()
