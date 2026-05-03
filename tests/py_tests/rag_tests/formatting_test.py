"""Tests for formatting helpers."""
import unittest

from langchain_core.documents import Document

from src.rag.formatting import build_default_retrieved_books_format


class TestFormatting(unittest.TestCase):
    def test_build_default_retrieved_books_format_happy_path(self):
        """Test happy path of formatting retrieved books"""
        books = [
            Document(
                page_content="Title: Book A\nAuthor: Alice\nGenre: Fantasy\nDescription: Indexed text",
                metadata={
                    "title": "Book A",
                    "author": "Alice",
                    "description": "Desc A",
                    "thumbnail": "https://example.com/a.jpg",
                    "categories": "Fantasy",
                },
            ),
            Document(
                page_content="Title: Book B\nDescription: Indexed text",
                metadata={"title": "Book B", "description": "Desc B"},
            ),
        ]

        formatted = build_default_retrieved_books_format(books)

        self.assertIn("Recommendation 1", formatted)
        self.assertIn("description: Desc A", formatted)
        self.assertIn("title: Book A", formatted)
        self.assertIn("authors: Alice", formatted)
        self.assertIn("thumbnail: https://example.com/a.jpg", formatted)
        self.assertIn("- categories: Fantasy", formatted)
        self.assertNotIn("- description:", formatted)
        self.assertIn("Recommendation 2", formatted)
        self.assertIn("description: Desc B", formatted)
        self.assertIn("title: Book B", formatted)
        self.assertIn("authors: N/A", formatted)
        self.assertIn("thumbnail: N/A", formatted)

    def test_build_default_retrieved_books_format_empty_list(self):
        """Test that empy books list raises ValueError."""
        with self.assertRaises(ValueError):
            build_default_retrieved_books_format([])

    def test_build_default_retrieved_books_format_invalid_item(self):
        """Test that non-Document items raise ValueError."""
        books = [Document(page_content="Desc A"), "not-doc"]

        with self.assertRaises(ValueError):
            build_default_retrieved_books_format(books)


if __name__ == "__main__":
    unittest.main()
