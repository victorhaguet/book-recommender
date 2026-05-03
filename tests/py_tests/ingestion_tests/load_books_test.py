import os
import tempfile
import unittest

import pandas as pd
from langchain_core.documents import Document

from src.ingestion.load_books import load_books


class TestBookLoader(unittest.TestCase):

    def setUp(self):
        """ Create a temporary CSV file for tests
        """
        self.tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        self.csv_path = self.tmp.name
        self.tmp.close()

    def tearDown(self):
        """ Cleanup the temp file
        """
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def _write_df_to_csv(self, df: pd.DataFrame) -> None:
        """ Create a CSV from a pandas dataframe
        """
        df.to_csv(self.csv_path, index=False)

    def test_load_books_happy_path(self):
        """ Happy path test
        """
        df = pd.DataFrame(
            [
                {
                    "title": "Book A",
                    "author": "Alice",
                    "genre": "Fantasy",
                    "description": "A gifted young hero uncovers a royal conspiracy that threatens his city forever.",
                    "year": 2020,
                },
                {
                    "title": "Book B",
                    "author": "Bob",
                    "description": "A quiet scholar is forced into court politics after finding forbidden letters.",
                    "year": 2021,
                },
            ]
        )
        self._write_df_to_csv(df)

        docs = load_books(self.csv_path, columns_to_drop=["year"])

        # Check that we have two LangChain documents (one for each book)
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(isinstance(d, Document) for d in docs))

        # Check contents and metadata
        self.assertEqual(
            docs[0].page_content,
            "\n".join(
                [
                    "Title: Book A",
                    "Author: Alice",
                    "Genre: Fantasy",
                    "Description: A gifted young hero uncovers a royal conspiracy that threatens his city forever.",
                ]
            ),
        )
        self.assertEqual(
            docs[0].metadata,
            {
                "title": "Book A",
                "author": "Alice",
                "genre": "Fantasy",
                "description": "A gifted young hero uncovers a royal conspiracy that threatens his city forever.",
            },
        )

        self.assertEqual(
            docs[1].page_content,
            "\n".join(
                [
                    "Title: Book B",
                    "Author: Bob",
                    "Genre: Unknown",
                    "Description: A quiet scholar is forced into court politics after finding forbidden letters.",
                ]
            ),
        )
        self.assertEqual(
            docs[1].metadata,
            {
                "title": "Book B",
                "author": "Bob",
                "genre": "Unknown",
                "description": "A quiet scholar is forced into court politics after finding forbidden letters.",
            },
        )

    def test_load_books_filters_missing_blank_and_short_descriptions(self):
        """ Test that books without usable descriptions are dropped.
        """
        df = pd.DataFrame(
            [
                {
                    "title": "Book C",
                    "description": "A reluctant heir investigates crimes inside the royal court to protect his home city.",
                },
                {"title": "Book D", "description": None},
                {"title": "Book E", "description": float("nan")},
                {"title": "Book F", "description": "   "},
                {"title": "Book G", "description": "fantasy novel"},
            ]
        )
        self._write_df_to_csv(df)

        docs = load_books(self.csv_path, columns_to_drop=[])

        self.assertEqual(len(docs), 1)
        self.assertEqual([d.metadata["title"] for d in docs], ["Book C"])
        self.assertEqual(
            docs[0].metadata["description"],
            "A reluctant heir investigates crimes inside the royal court to protect his home city.",
        )

    def test_load_books_drop_columns(self):
        """ Test whether the columns have been dropped. 
        """
        df = pd.DataFrame(
            [
                {
                    "title": "Book G",
                    "author": "Alice",
                    "genre": "Sci-Fi",
                    "description": "An isolated pilot survives on a hostile planet while decoding an alien distress beacon.",
                },
            ]
        )
        self._write_df_to_csv(df)

        docs = load_books(self.csv_path, columns_to_drop=["genre"])

        self.assertEqual(len(docs), 1)
        self.assertEqual(
            docs[0].metadata,
            {
                "title": "Book G",
                "author": "Alice",
                "description": "An isolated pilot survives on a hostile planet while decoding an alien distress beacon.",
            },
        )
        self.assertNotIn("genre", docs[0].metadata)

    def test_load_books_raises_if_description_column_missing(self):
        """ Test if the code raise an error in case a book is missing description
        """
        df = pd.DataFrame([{"title": "Book H", "author": "Alice"}])
        self._write_df_to_csv(df)

        with self.assertRaises(KeyError):
            load_books(self.csv_path, columns_to_drop=[])

    def test_load_books_invalid_drop_column_raises(self):
        """ Test for errors if an attempt is made to drop a non-existent column.
        """
        df = pd.DataFrame(
            [
                {"title": "Book I", "description": "Desc I"},
            ]
        )
        self._write_df_to_csv(df)

        # pandas.DataFrame.drop raises KeyError by default for unknown columns
        with self.assertRaises(ValueError):
            load_books(self.csv_path, columns_to_drop=["does_not_exist"])

    def test_load_books_rejects_invalid_min_description_words(self):
        """Test that invalid minimum word counts raise a ValueError."""
        df = pd.DataFrame(
            [
                {"title": "Book I", "description": "This valid description contains more than enough words for indexing."},
            ]
        )
        self._write_df_to_csv(df)

        with self.assertRaises(ValueError):
            load_books(self.csv_path, columns_to_drop=[], min_description_words=-1)

    def test_load_books_raises_for_missing_dataset_file(self):
        """ Test that a FileNotFoundError is raised when the specified CSV file does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_books("/tmp/definitely-missing-books.csv", columns_to_drop=[])

    def test_load_books_raises_for_invalid_csv_content(self):
        """ Test that a ValueError is raised when the CSV content is invalid."""
        with open(self.csv_path, "w", encoding="utf-8") as handle:
            handle.write('title,description\n"broken')

        with self.assertRaises(ValueError):
            load_books(self.csv_path, columns_to_drop=[])

    def test_load_books_rejects_dropping_description_column(self):
        """ Test that a ValueError is raised if the user tries to drop the description column, which is required for the page_content of the documents."""
        df = pd.DataFrame(
            [
                {"title": "Book I", "description": "Desc I"},
            ]
        )
        self._write_df_to_csv(df)

        with self.assertRaises(ValueError):
            load_books(self.csv_path, columns_to_drop=["description"])

if __name__ == "__main__":
    unittest.main()
