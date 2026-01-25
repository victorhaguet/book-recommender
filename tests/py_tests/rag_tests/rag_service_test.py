"""Tests for RAGService."""
import unittest
from unittest.mock import MagicMock, patch

from src.rag.rag_service import RAGService


class TestRAGService(unittest.TestCase):
    @patch("src.rag.rag_service.build_chain")
    @patch("src.rag.rag_service.build_default_retrieved_books_format")
    @patch("src.rag.rag_service.get_default_generation_prompt")
    @patch("src.rag.rag_service.set_retrieving_strategy")
    def test_init_builds_chain_with_components(
        self,
        mock_set_retrieving_strategy,
        mock_get_prompt,
        mock_build_format,
        mock_build_chain,
    ):
        """Test that init builds chain with correct components."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        retriever = MagicMock(name="Retriever")
        prompt = MagicMock(name="Prompt")
        chain = MagicMock(name="Chain")

        mock_set_retrieving_strategy.return_value = retriever
        mock_get_prompt.return_value = prompt
        mock_build_chain.return_value = chain

        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)

        self.assertIs(service.llm, llm)
        self.assertIs(service.retriever, retriever)
        self.assertIs(service.prompt, prompt)
        self.assertIs(service.format, mock_build_format)
        self.assertIs(service.chain, chain)
        mock_set_retrieving_strategy.assert_called_once_with(vectorstore=vectorstore, k=3)
        mock_get_prompt.assert_called_once_with()
        mock_build_chain.assert_called_once_with(
            retriever=retriever,
            llm=llm,
            prompt=prompt,
            format_docs=mock_build_format,
        )

    def test_init_validates_inputs(self):
        """Test that invalid inputs raise ValueErrors."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")

        with self.assertRaises(ValueError):
            RAGService(llm=None, vectorstore=vectorstore, k=3)
        with self.assertRaises(ValueError):
            RAGService(llm=llm, vectorstore=None, k=3)
        with self.assertRaises(ValueError):
            RAGService(llm=llm, vectorstore=vectorstore, k=0)
        with self.assertRaises(ValueError):
            RAGService(llm=llm, vectorstore=vectorstore, k=-1)
        with self.assertRaises(ValueError):
            RAGService(llm=llm, vectorstore=vectorstore, k="3")

    @patch("src.rag.rag_service.build_chain")
    @patch("src.rag.rag_service.set_retrieving_strategy")
    def test_answer_query_delegates_to_chain(
        self,
        mock_set_retrieving_strategy,
        mock_build_chain,
    ):
        """Test that answer_query calls the chain with the query."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        retriever = MagicMock(name="Retriever")
        chain = MagicMock(name="Chain")
        chain.return_value = {"response": "ok", "retrieved_documents": []}

        mock_set_retrieving_strategy.return_value = retriever
        mock_build_chain.return_value = chain

        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)
        result = service.answer_query("hello")

        self.assertEqual(result, chain.return_value)
        chain.assert_called_once_with("hello")

    @patch("src.rag.rag_service.build_chain")
    @patch("src.rag.rag_service.set_retrieving_strategy")
    def test_answer_query_invalid_query(self, mock_set_retrieving_strategy, mock_build_chain):
        """Test that invalid queries raise ValueErrors."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        mock_set_retrieving_strategy.return_value = MagicMock(name="Retriever")
        mock_build_chain.return_value = MagicMock(name="Chain")
        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)

        for value in ("", None, 123):
            with self.assertRaises(ValueError):
                service.answer_query(value)

    @patch("src.rag.rag_service.build_chain")
    @patch("src.rag.rag_service.set_retrieving_strategy")
    def test_answer_query_chain_error_wrapped(
        self,
        mock_set_retrieving_strategy,
        mock_build_chain,
    ):
        """Test that chain errors are wrapped."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        retriever = MagicMock(name="Retriever")
        chain = MagicMock(name="Chain")
        chain.side_effect = RuntimeError("boom")

        mock_set_retrieving_strategy.return_value = retriever
        mock_build_chain.return_value = chain

        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)

        with self.assertRaises(RuntimeError):
            service.answer_query("hello")


if __name__ == "__main__":
    unittest.main()
