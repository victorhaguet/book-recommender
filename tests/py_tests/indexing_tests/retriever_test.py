"""Tests for retriever helpers."""
import unittest
from unittest.mock import MagicMock

from src.indexing.retriever import set_retrieving_strategy


class TestRetrieverHelpers(unittest.TestCase):
    def test_set_retrieving_strategy_calls_vectorstore(self):
        """Creating retriever delegates to vectorstore."""
        vectorstore = MagicMock(name="FAISSStore")
        fake_retriever = MagicMock(name="Retriever")
        vectorstore.as_retriever.return_value = fake_retriever

        retriever = set_retrieving_strategy(vectorstore, k=3)

        self.assertIs(retriever, fake_retriever)
        vectorstore.as_retriever.assert_called_once_with(k=3)

    def test_set_retrieving_strategy_invalid_vectorstore(self):
        """None vectorstore should fail."""
        with self.assertRaises(ValueError):
            set_retrieving_strategy(None, k=3)

    def test_set_retrieving_strategy_invalid_k(self):
        """Invalid k should fail."""
        vectorstore = MagicMock(name="FAISSStore")

        for value in (0, -1, 1.5, "3", None):
            with self.assertRaises(ValueError):
                set_retrieving_strategy(vectorstore, k=value)

    def test_set_retrieving_strategy_error_wrapped(self):
        """Vectorstore errors should be wrapped."""
        vectorstore = MagicMock(name="FAISSStore")
        vectorstore.as_retriever.side_effect = RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            set_retrieving_strategy(vectorstore, k=3)

if __name__ == "__main__":
    unittest.main()
