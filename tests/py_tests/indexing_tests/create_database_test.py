"""Tests for create_database helper."""
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.indexing.create_database import create_database


DATA_PATH="/tmp/books.csv"
INDEX_PATH="/tmp/index"
INDEX_NAME="books_index"

class TestCreateDatabase(unittest.TestCase):
    """
    Test create database function
    """
    @patch("src.indexing.create_database.save_store")
    @patch("src.indexing.create_database.add_documents")
    @patch("src.indexing.create_database.create_faiss_db")
    @patch("src.indexing.create_database.load_books")
    def test_create_database_happy_path(
        self,
        mock_load_books,
        mock_create_faiss_db,
        mock_add_documents,
        mock_save_store,
    ):
        """Test happy path"""
        embeddings = MagicMock(name="Embeddings")
        docs = [Document(page_content="a"), Document(page_content="b")]
        vectorstore = MagicMock(name="FAISSStore")

        mock_load_books.return_value = docs
        mock_create_faiss_db.return_value = vectorstore

        result = create_database(
            data_path=DATA_PATH,
            columns_to_drop=["col1"],
            min_description_words=10,
            embeddings=embeddings,
            index_path=INDEX_PATH,
            index_name=INDEX_NAME,
        )

        self.assertIs(result, vectorstore)
        mock_load_books.assert_called_once_with(
            path=DATA_PATH,
            columns_to_drop=["col1"],
            min_description_words=10,
        )
        mock_create_faiss_db.assert_called_once_with(embeddings)
        mock_add_documents.assert_called_once_with(
            documents=docs,
            vectorstore=vectorstore,
        )
        mock_save_store.assert_called_once_with(
            vectorstore=vectorstore,
            path=INDEX_PATH,
            index_name=INDEX_NAME,
        )

    @patch("src.indexing.create_database.save_store")
    @patch("src.indexing.create_database.add_documents")
    @patch("src.indexing.create_database.create_faiss_db")
    @patch("src.indexing.create_database.load_books")
    def test_create_database_load_books_error_propagates(
        self,
        mock_load_books,
        mock_create_faiss_db,
        mock_add_documents,
        mock_save_store,
    ):
        """Test error propagation from load_books()"""
        mock_load_books.side_effect = ValueError("bad csv")

        with self.assertRaises(ValueError):
            create_database(
                data_path=DATA_PATH,
                columns_to_drop=["col1"],
                min_description_words=10,
                embeddings=MagicMock(name="Embeddings"),
                index_path=INDEX_PATH,
                index_name=INDEX_NAME,
            )

        mock_create_faiss_db.assert_not_called()
        mock_add_documents.assert_not_called()
        mock_save_store.assert_not_called()

    @patch("src.indexing.create_database.save_store")
    @patch("src.indexing.create_database.add_documents")
    @patch("src.indexing.create_database.create_faiss_db")
    @patch("src.indexing.create_database.load_books")
    def test_create_database_create_faiss_db_error_propagates(
        self,
        mock_load_books,
        mock_create_faiss_db,
        mock_add_documents,
        mock_save_store,
    ):
        """Test error propagation from faiss vectorstore creation"""
        mock_load_books.return_value = [Document(page_content="a")]
        mock_create_faiss_db.side_effect = RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            create_database(
                data_path=DATA_PATH,
                columns_to_drop=["col1"],
                min_description_words=10,
                embeddings=MagicMock(name="Embeddings"),
                index_path=INDEX_PATH,
                index_name=INDEX_NAME,
            )

        mock_add_documents.assert_not_called()
        mock_save_store.assert_not_called()

    @patch("src.indexing.create_database.save_store")
    @patch("src.indexing.create_database.add_documents")
    @patch("src.indexing.create_database.create_faiss_db")
    @patch("src.indexing.create_database.load_books")
    def test_create_database_add_documents_error_propagates(
        self,
        mock_load_books,
        mock_create_faiss_db,
        mock_add_documents,
        mock_save_store,
    ):
        """Test error propagation when error is raised from the add_documents function"""
        vectorstore = MagicMock(name="FAISSStore")
        mock_load_books.return_value = [Document(page_content="a")]
        mock_create_faiss_db.return_value = vectorstore
        mock_add_documents.side_effect = ValueError("bad docs")

        with self.assertRaises(ValueError):
            create_database(
                data_path=DATA_PATH,
                columns_to_drop=["col1"],
                min_description_words=10,
                embeddings=MagicMock(name="Embeddings"),
                index_path=INDEX_PATH,
                index_name=INDEX_NAME,
            )

        mock_save_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
