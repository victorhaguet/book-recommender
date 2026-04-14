"""Tests for the LangGraph-backed agentic RAG service."""

import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import quote_plus

from src.rag.agentic_service import (
    AgenticRAGService,
    DEFAULT_CLARIFICATION_MESSAGE,
    DEFAULT_PROBLEMATIC_MESSAGE,
    DEFAULT_SCOPE_MESSAGE,
    QueryRecognitionResult,
    QueryRoute,
)


class _StructuredRecognizer:
    def __init__(self, result):
        self._result = result

    def invoke(self, _messages):
        return self._result


class TestAgenticRAGService(unittest.TestCase):
    @patch("src.rag.agentic_service.RAGService")
    def test_init_validates_inputs(self, mock_rag_service_cls):
        """Test that the AgenticRAGService constructor raises ValueError when given invalid inputs."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        llm = MagicMock(name="LLM")
        vectorstore = MagicMock(name="Vectorstore")

        with self.assertRaises(ValueError):
            AgenticRAGService(llm=None, vectorstore=vectorstore, k=3)
        with self.assertRaises(ValueError):
            AgenticRAGService(llm=llm, vectorstore=None, k=3)
        with self.assertRaises(ValueError):
            AgenticRAGService(llm=llm, vectorstore=vectorstore, k=0)
        with self.assertRaises(ValueError):
            AgenticRAGService(llm=llm, vectorstore=vectorstore, k=-1)
        with self.assertRaises(ValueError):
            AgenticRAGService(llm=llm, vectorstore=vectorstore, k="3")

    @patch("src.rag.agentic_service.RAGService")
    def test_answer_query_validates_inputs(self, mock_rag_service_cls):
        """Test that the answer_query method raises ValueError when given invalid inputs."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        with self.assertRaises(ValueError):
            service.answer_query("", "thread-1")
        with self.assertRaises(ValueError):
            service.answer_query("hello", "")

    @patch("src.rag.agentic_service.RAGService")
    @patch("src.rag.agentic_service.dotenv_values")
    def test_get_dotenv_value_handles_missing_blank_and_value(self, mock_dotenv_values, mock_rag_service_cls):
        """Test that the _get_dotenv_value method correctly handles missing, blank, and valid values from dotenv."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        mock_dotenv_values.return_value = {}
        self.assertIsNone(service._get_dotenv_value("TAVILY_API_KEY"))

        mock_dotenv_values.return_value = {"TAVILY_API_KEY": "   "}
        self.assertIsNone(service._get_dotenv_value("TAVILY_API_KEY"))

        mock_dotenv_values.return_value = {"TAVILY_API_KEY": " secret "}
        self.assertEqual(service._get_dotenv_value("TAVILY_API_KEY"), "secret")

    @patch("src.rag.agentic_service.RAGService")
    def test_new_recommendation_route_uses_recommendation_service(self, mock_rag_service_cls):
        """Test that a query recognized as a new recommendation request is processed through the RAGService and formatted correctly."""
        llm = MagicMock(name="LLM")
        llm.with_structured_output.return_value = _StructuredRecognizer(
            QueryRecognitionResult(
                route=QueryRoute.NEW_RECOMMENDATION,
                retrieval_query="epic fantasy",
                referenced_items=[],
            )
        )
        mock_rag_service = MagicMock(name="RAGService")
        mock_rag_service.answer_query.return_value = {
            "response": "Here are some recommendations.",
            "recommendations": [
                {"title": "Book A", "author": "Alice", "summary": "Summary", "thumbnail": None}
            ],
        }
        mock_rag_service_cls.return_value = mock_rag_service

        service = AgenticRAGService(llm=llm, vectorstore=MagicMock(), k=3)
        result = service.answer_query("I want fantasy", "thread-1")

        self.assertIn("If one of these books interests you", result["response"])
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["sources"], [])
        mock_rag_service.answer_query.assert_called_once_with("epic fantasy")

    @patch("src.rag.agentic_service.RAGService")
    def test_irrelevant_query_returns_scope_message(self, mock_rag_service_cls):
        """Test that a query recognized as irrelevant returns the default scope message."""
        llm = MagicMock(name="LLM")
        llm.with_structured_output.return_value = _StructuredRecognizer(
            QueryRecognitionResult(
                route=QueryRoute.IRRELEVANT,
                retrieval_query="",
                referenced_items=[],
            )
        )
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")

        service = AgenticRAGService(llm=llm, vectorstore=MagicMock(), k=3)
        result = service.answer_query("What is the weather?", "thread-1")

        self.assertEqual(result["response"], DEFAULT_SCOPE_MESSAGE)
        self.assertEqual(result["recommendations"], [])

    @patch("src.rag.agentic_service.RAGService")
    def test_ambiguous_query_returns_clarification(self, mock_rag_service_cls):
        """Test that a query recognized as ambiguous returns the default clarification message."""
        llm = MagicMock(name="LLM")
        llm.with_structured_output.return_value = _StructuredRecognizer(
            QueryRecognitionResult(
                route=QueryRoute.AMBIGUOUS,
                retrieval_query="",
                referenced_items=[],
            )
        )
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")

        service = AgenticRAGService(llm=llm, vectorstore=MagicMock(), k=3)
        result = service.answer_query("tell me more about the first one or maybe another one", "thread-1")

        self.assertEqual(result["response"], DEFAULT_CLARIFICATION_MESSAGE)

    @patch("src.rag.agentic_service.RAGService")
    def test_follow_up_route_uses_parallel_lookups_and_returns_sources(self, mock_rag_service_cls):
        """Test that a query recognized as a follow-up triggers parallel lookups and returns a response with merged sources."""
        llm = MagicMock(name="LLM")
        llm.with_structured_output.side_effect = [
            _StructuredRecognizer(
                QueryRecognitionResult(
                    route=QueryRoute.NEW_RECOMMENDATION,
                    retrieval_query="epic fantasy",
                    referenced_items=[],
                )
            ),
            _StructuredRecognizer(
                QueryRecognitionResult(
                    route=QueryRoute.FOLLOW_UP,
                    retrieval_query="",
                    referenced_items=["Book A", "Alice"],
                )
            ),
        ]
        llm.invoke.side_effect = [
            MagicMock(content="Background note"),
            MagicMock(content="Follow-up answer with sources listed below."),
        ]
        mock_rag_service = MagicMock(name="RAGService")
        mock_rag_service.answer_query.return_value = {
            "response": "Here are some recommendations.",
            "recommendations": [
                {"title": "Book A", "author": "Alice", "summary": "Summary", "thumbnail": None}
            ],
        }
        mock_rag_service_cls.return_value = mock_rag_service

        service = AgenticRAGService(llm=llm, vectorstore=MagicMock(), k=3)
        service.answer_query("I want fantasy", "thread-1")

        with patch.object(
            service,
            "_run_tavily_search",
            return_value={"summary": "Tavily", "sources": [{"title": "Source A", "url": "https://a"}]},
        ), patch.object(
            service,
            "_run_wikipedia_search",
            return_value={"summary": "Wiki", "sources": [{"title": "Wiki", "url": "https://wiki"}]},
        ):
            result = service.answer_query("Tell me more about the author", "thread-1")

        self.assertEqual(result["recommendations"], [])
        self.assertEqual(len(result["sources"]), 2)
        self.assertIn("Follow-up answer", result["response"])

    @patch("src.rag.agentic_service.RAGService")
    def test_reject_query_node_returns_problematic_message(self, mock_rag_service_cls):
        """Test that a query recognized as problematic returns the default problematic message."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        result = service._reject_query_node({"route": QueryRoute.PROBLEMATIC.value})

        self.assertEqual(result["response"], DEFAULT_PROBLEMATIC_MESSAGE)

    @patch("src.rag.agentic_service.RAGService")
    def test_recognize_query_node_resets_history_for_new_recommendation(self, mock_rag_service_cls):
        """Test that a query recognized as a new recommendation request resets the recommendation history and formats the reset summary appropriately."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        with patch.object(
            service,
            "_recognize_query",
            return_value=QueryRecognitionResult(
                route=QueryRoute.NEW_RECOMMENDATION,
                retrieval_query="fresh query",
                referenced_items=[],
            ),
        ):
            result = service._recognize_query_node(
                {
                    "user_query": "new request",
                    "has_history": True,
                    "recommended_books": [{"title": "Old Book", "author": "Old Author"}],
                    "recommended_books_summary": "old summary",
                    "messages": [],
                }
            )

        self.assertEqual(result["route"], QueryRoute.NEW_RECOMMENDATION.value)
        self.assertEqual(result["retrieval_query"], "fresh query")
        self.assertEqual(result["recommended_books"], [])
        self.assertIn("Previous recommendation thread reset", result["recommended_books_summary"])

    @patch("src.rag.agentic_service.RAGService")
    def test_build_reset_summary_prefers_existing_summary_without_books(self, mock_rag_service_cls):
        """Test that the _build_reset_summary method returns the existing summary if there are no recommended books, and only returns the default reset message if there is also no existing summary."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        summary = service._build_reset_summary(
            {"recommended_books": [], "recommended_books_summary": "kept summary"}
        )

        self.assertEqual(summary, "kept summary")

    @patch("src.rag.agentic_service.RAGService")
    def test_build_reset_summary_returns_default_without_books_or_summary(self, mock_rag_service_cls):
        """Test that the _build_reset_summary method returns the default reset message if there are no recommended books and no existing summary."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        summary = service._build_reset_summary({"recommended_books": [], "recommended_books_summary": ""})

        self.assertEqual(summary, "Previous recommendation thread reset after a new request.")

    @patch("src.rag.agentic_service.RAGService")
    def test_build_follow_up_search_query_uses_recent_messages_then_books_then_query(self, mock_rag_service_cls):
        """Test that the _build_follow_up_search_query method prioritizes recent messages, then recommended books, then the user query."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        by_recent_messages = service._build_follow_up_search_query(
            {
                "user_query": "Did she write a lot?",
                "referenced_items": [],
                "messages": [
                    {"role": "user", "content": "Who is Miriam Toews?"},
                    {"role": "assistant", "content": "Miriam Toews is a Canadian author."},
                ],
                "recommended_books": [],
            }
        )
        self.assertIn("Who is Miriam Toews?", by_recent_messages)

        by_books = service._build_follow_up_search_query(
            {
                "user_query": "tell me more",
                "referenced_items": [],
                "messages": [],
                "recommended_books": [
                    {"title": "Book A", "author": "Alice"},
                    {"title": "Book B", "author": "Bob"},
                ],
            }
        )
        self.assertIn("Book A Alice", by_books)

        plain = service._build_follow_up_search_query(
            {"user_query": "plain", "referenced_items": [], "messages": [], "recommended_books": []}
        )
        self.assertEqual(plain, "plain")

    @patch("src.rag.agentic_service.RAGService")
    @patch("src.rag.agentic_service.TavilySearch")
    def test_run_tavily_search_handles_skip_exception_and_response_shapes(
        self,
        mock_tavily_search,
        mock_rag_service_cls,
    ):
        """Test that the _run_tavily_search method correctly handles missing API key, exceptions from the search, and various response formats, always returning a dict with 'summary' and 'sources' keys."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        with patch.object(service, "_get_dotenv_value", return_value=None):
            self.assertEqual(service._run_tavily_search("query"), {"summary": "", "sources": []})

        with patch.object(service, "_get_dotenv_value", return_value="key"):
            mock_tavily_search.return_value.invoke.side_effect = RuntimeError("boom")
            self.assertEqual(service._run_tavily_search("query"), {"summary": "", "sources": []})

        with patch.object(service, "_get_dotenv_value", return_value="key"):
            mock_tavily_search.return_value.invoke.side_effect = None
            mock_tavily_search.return_value.invoke.return_value = (
                "summary",
                {"results": [{"title": "A", "url": "https://a"}, {"title": "B", "url": ""}]},
            )
            result = service._run_tavily_search("query")
            self.assertEqual(result["summary"], "summary")
            self.assertEqual(result["sources"], [{"title": "A", "url": "https://a"}])

            mock_tavily_search.return_value.invoke.return_value = {
                "content": "dict summary",
                "results": [{"title": "B", "url": "https://b"}],
            }
            result = service._run_tavily_search("query")
            self.assertEqual(result["summary"], "dict summary")
            self.assertEqual(result["sources"], [{"title": "B", "url": "https://b"}])

            mock_tavily_search.return_value.invoke.return_value = [
                {"content": "first", "title": "C", "url": "https://c"},
                {"content": "second", "title": "D", "url": ""},
            ]
            result = service._run_tavily_search("query")
            self.assertIn("first", result["summary"])
            self.assertEqual(result["sources"], [{"title": "C", "url": "https://c"}])

            mock_tavily_search.return_value.invoke.return_value = "raw text"
            result = service._run_tavily_search("query")
            self.assertEqual(result, {"summary": "raw text", "sources": []})

    @patch("src.rag.agentic_service.RAGService")
    @patch("src.rag.agentic_service.WikipediaAPIWrapper")
    def test_run_wikipedia_search_handles_exception_empty_and_success(
        self,
        mock_wrapper_cls,
        mock_rag_service_cls,
    ):
        """Test that the _run_wikipedia_search method correctly handles exceptions from the wrapper, empty responses, and successful responses, always returning a dict with 'summary' and 'sources' keys."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)
        wrapper = MagicMock(name="WikipediaWrapper")
        mock_wrapper_cls.return_value = wrapper

        wrapper.run.side_effect = RuntimeError("boom")
        self.assertEqual(service._run_wikipedia_search("query"), {"summary": "", "sources": []})

        wrapper.run.side_effect = None
        wrapper.run.return_value = ""
        self.assertEqual(service._run_wikipedia_search("query"), {"summary": "", "sources": []})

        wrapper.run.return_value = "wiki summary"
        result = service._run_wikipedia_search("Margaret Atwood")
        self.assertEqual(result["summary"], "wiki summary")
        self.assertEqual(
            result["sources"],
            [
                {
                    "title": "Wikipedia search: Margaret Atwood",
                    "url": "https://en.wikipedia.org/wiki/Special:Search?search="
                    + quote_plus("Margaret Atwood"),
                }
            ],
        )

    @patch("src.rag.agentic_service.RAGService")
    def test_merge_sources_skips_invalid_empty_and_duplicate_urls(self, mock_rag_service_cls):
        """Test that the _merge_sources method correctly merges two lists of sources while skipping invalid entries, empty URLs, and duplicates based on URL."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        merged = service._merge_sources(
            [
                "bad",
                {"title": "A", "url": "https://a"},
                {"title": "A again", "url": "https://a"},
                {"title": "No URL", "url": ""},
            ],
            [
                {"title": "B", "url": "https://b"},
            ],
        )

        self.assertEqual(
            merged,
            [
                {"title": "A", "url": "https://a"},
                {"title": "B", "url": "https://b"},
            ],
        )

    @patch("src.rag.agentic_service.RAGService")
    def test_persist_thread_state_handles_follow_up_and_keeps_existing_recommendations(self, mock_rag_service_cls):
        """Test that the _persist_thread_state method correctly handles follow-up queries and keeps existing recommendations."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)
        service._threads["thread-1"] = {
            "messages": [{"role": "user", "content": "old"}],
            "recommended_books": [{"title": "Book A", "author": "Alice"}],
            "recommended_books_summary": "",
        }

        service._persist_thread_state(
            thread_id="thread-1",
            user_query="tell me more",
            response="response",
            recommendations=[],
            previous_messages=[{"role": "user", "content": "old"}],
            previous_summary="",
            route=QueryRoute.FOLLOW_UP.value,
        )

        snapshot = service._threads["thread-1"]
        self.assertEqual(snapshot["recommended_books"], [{"title": "Book A", "author": "Alice"}])
        self.assertEqual(snapshot["recommended_books_summary"], "")
        self.assertEqual(len(snapshot["messages"]), 3)

    @patch("src.rag.agentic_service.RAGService")
    def test_build_summary_from_recommendations_and_format_recent_messages(self, mock_rag_service_cls):
        """Test that the _build_summary_from_recommendations method formats a summary of recommended books correctly, and that the _format_recent_messages method formats recent messages while handling missing roles."""
        mock_rag_service_cls.return_value = MagicMock(name="RAGService")
        service = AgenticRAGService(llm=MagicMock(name="LLM"), vectorstore=MagicMock(), k=3)

        self.assertEqual(service._build_summary_from_recommendations([]), "")
        self.assertEqual(
            service._build_summary_from_recommendations(
                [{"title": "Book A", "author": "Alice"}, {"title": "Book B", "author": "Bob"}]
            ),
            "Most recent recommended books: Book A by Alice; Book B by Bob.",
        )

        formatted = service._format_recent_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": ""},
                {"role": "", "content": "fallback role"},
            ]
        )
        self.assertEqual(formatted, ["user: hello", "unknown: fallback role"])
