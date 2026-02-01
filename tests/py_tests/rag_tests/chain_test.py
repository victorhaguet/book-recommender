"""Tests for RAG chain builder."""
import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage

from src.rag.chain import build_chain


class TestBuildChain(unittest.TestCase):
    def _make_happy_deps(self):
        """ Create mocks for happy path tests."""
        retriever = MagicMock(name="Retriever")
        retriever.invoke.return_value = [Document(page_content="Doc 1")]

        llm = MagicMock(name="ChatOpenAI")
        llm.invoke.return_value = AIMessage(content="Answer")

        prompt = MagicMock(name="ChatPromptTemplate")
        prompt.format_messages.return_value = [MagicMock(spec=BaseMessage)]

        format_docs = MagicMock(name="format_docs")
        format_docs.return_value = "formatted"

        return retriever, llm, prompt, format_docs

    def test_build_chain_happy_path(self):
        """Test the happy path of the chain."""
        retriever, llm, prompt, format_docs = self._make_happy_deps()
        chain = build_chain(retriever, llm, prompt, format_docs)

        result = chain("query")

        self.assertEqual(result["response"], "Answer")
        self.assertEqual(result["retrieved_documents"], retriever.invoke.return_value)
        retriever.invoke.assert_called_once_with(input="query")
        format_docs.assert_called_once_with(retriever.invoke.return_value)
        prompt.format_messages.assert_called_once_with(retrieved_books="formatted")
        llm.invoke.assert_called_once_with(input=prompt.format_messages.return_value)

    def test_build_chain_validates_inputs(self):
        """Check that invalid inputs raise ValueErrors."""
        retriever, llm, prompt, format_docs = self._make_happy_deps()

        with self.assertRaises(ValueError):
            build_chain(None, llm, prompt, format_docs)
        with self.assertRaises(ValueError):
            build_chain(retriever, None, prompt, format_docs)
        with self.assertRaises(ValueError):
            build_chain(retriever, llm, None, format_docs)
        with self.assertRaises(ValueError):
            build_chain(retriever, llm, prompt, None)
        with self.assertRaises(ValueError):
            build_chain(retriever, llm, prompt, "not-callable")

    def test_chain_retriever_error_wrapped(self):
        """Check that retriever errors are wrapped."""
        retriever, llm, prompt, format_docs = self._make_happy_deps()
        retriever.invoke.side_effect = RuntimeError("boom")
        chain = build_chain(retriever, llm, prompt, format_docs)

        with self.assertRaises(RuntimeError):
            chain("query")

    def test_chain_format_docs_error_wrapped(self):
        """Check that format_docs errors are wrapped."""
        retriever, llm, prompt, format_docs = self._make_happy_deps()
        format_docs.side_effect = RuntimeError("boom")
        chain = build_chain(retriever, llm, prompt, format_docs)

        with self.assertRaises(RuntimeError):
            chain("query")

    def test_chain_prompt_format_error_wrapped(self):
        """Check that prompt formatting errors are wrapped."""
        retriever, llm, prompt, format_docs = self._make_happy_deps()
        prompt.format_messages.side_effect = RuntimeError("boom")
        chain = build_chain(retriever, llm, prompt, format_docs)

        with self.assertRaises(RuntimeError):
            chain("query")

    def test_chain_llm_error_wrapped(self):
        """Check that LLM invocation errors are wrapped."""
        retriever, llm, prompt, format_docs = self._make_happy_deps()
        llm.invoke.side_effect = RuntimeError("boom")
        chain = build_chain(retriever, llm, prompt, format_docs)

        with self.assertRaises(RuntimeError):
            chain("query")


if __name__ == "__main__":
    unittest.main()
