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
                {"title": "Book A", "author": "Alice", "description": "Desc A", "year": 2020},
                {"title": "Book B", "author": "Bob", "description": "Desc B", "year": 2021},
            ]
        )
        self._write_df_to_csv(df)

        docs = load_books(self.csv_path, columns_to_drop=["year"])

        # Check that we have two LangChain documents (one for each book)
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(isinstance(d, Document) for d in docs))

        # Check contents and metadata
        self.assertEqual(docs[0].page_content, "Desc A")
        self.assertEqual(docs[0].metadata, {"title": "Book A", "author": "Alice"})

        self.assertEqual(docs[1].page_content, "Desc B")
        self.assertEqual(docs[1].metadata, {"title": "Book B", "author": "Bob"})

    def test_load_books_filters_missing_descriptions(self):
        """ Test that books without descriptions are dropped.
        """
        df = pd.DataFrame(
            [
                {"title": "Book C", "description": "Desc C"},
                {"title": "Book D", "description": None},
                {"title": "Book E", "description": float("nan")},
                {"title": "Book F", "description": "Desc F"},
            ]
        )
        self._write_df_to_csv(df)

        docs = load_books(self.csv_path, columns_to_drop=[])

        # Ensure that only books with an STR value in the description field are kept.
        self.assertEqual(len(docs), 2)
        self.assertEqual([d.metadata["title"] for d in docs], ["Book C", "Book F"])
        self.assertEqual([d.page_content for d in docs], ["Desc C", "Desc F"])

    def test_load_books_drop_columns(self):
        """ Test whether the columns have been dropped. 
        """
        df = pd.DataFrame(
            [
                {"title": "Book G", "author": "Alice", "genre": "Sci-Fi", "description": "Desc G"},
            ]
        )
        self._write_df_to_csv(df)

        docs = load_books(self.csv_path, columns_to_drop=["genre"])

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata, {"title": "Book G", "author": "Alice"})
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
