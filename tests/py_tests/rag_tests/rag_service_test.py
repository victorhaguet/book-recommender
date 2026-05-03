"""Tests for RAGService."""
import json
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.rag_service import (
    RAGService,
    _build_fallback_recommendation_cards,
    _build_recommendation_cards,
    _extract_json_payload,
    _stringify,
)


class TestRAGService(unittest.TestCase):
    def test_stringify_handles_none_blank_and_values(self):
        """Test that _stringify converts None and blank strings to "N/A", and leaves non-blank strings unchanged."""
        self.assertEqual(_stringify(None), "N/A")
        self.assertEqual(_stringify("   "), "N/A")
        self.assertEqual(_stringify("hello"), "hello")

    def test_extract_json_payload_supports_markdown_fences(self):
        """Test that _extract_json_payload can extract JSON from markdown code fences."""
        payload = _extract_json_payload(
            '```json\n{"intro": "Hello", "recommendations": []}\n```'
        )

        self.assertEqual(payload, {"intro": "Hello", "recommendations": []})

    def test_extract_json_payload_requires_json_object(self):
        """Test that _extract_json_payload raises a ValueError if the JSON is not an object."""
        with self.assertRaises(ValueError):
            _extract_json_payload('["not", "an", "object"]')

    def test_build_recommendation_cards_requires_list_payload(self):
        """Test that _build_recommendation_cards raises a ValueError if the recommendations payload is not a list."""
        with self.assertRaises(ValueError):
            _build_recommendation_cards({"recommendations": "bad"}, [])

    def test_build_recommendation_cards_skips_invalid_items_and_out_of_range_ranks(self):
        """Test that _build_recommendation_cards skips items that are not dicts, that do not have a valid recommendation_number, or whose recommendation_number is out of range relative to the retrieved documents."""
        docs = [
            Document(
                page_content="Desc A",
                metadata={"title": "Book A", "author": "Alice", "num_pages": 123},
            )
        ]

        cards = _build_recommendation_cards(
            {
                "recommendations": [
                    "bad-item",
                    {"recommendation_number": "1", "summary": "bad rank type"},
                    {"recommendation_number": 2, "summary": "out of range"},
                    {"recommendation_number": 1, "summary": "Strong match"},
                ]
            },
            docs,
        )

        self.assertEqual(
            cards,
            [
                {
                    "title": "Book A",
                    "author": "Alice",
                    "summary": "Strong match",
                    "thumbnail": None,
                    "num_pages": 123,
                }
            ],
        )

    def test_build_fallback_recommendation_cards_truncates_long_descriptions(self):
        """Test that _build_fallback_recommendation_cards truncates long descriptions to 280 characters and appends ellipses."""
        docs = [
            Document(
                page_content="A" * 300,
                metadata={
                    "title": "Book A",
                    "authors": "Alice",
                    "description": "A" * 300,
                    "thumbnail": "",
                    "num_pages": 123,
                },
            ),
            Document(
                page_content="Title: Book B\nDescription: Indexed text",
                metadata={
                    "title": "Book B",
                    "author": "Bob",
                    "description": "Desc B",
                    "thumbnail": "https://example.com/b.jpg",
                },
            ),
        ]

        cards = _build_fallback_recommendation_cards(docs, limit=1)

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], "Book A")
        self.assertEqual(cards[0]["author"], "Alice")
        self.assertTrue(cards[0]["summary"].endswith("..."))
        self.assertEqual(len(cards[0]["summary"]), 280)
        self.assertIsNone(cards[0]["thumbnail"])

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
        chain.return_value = {
            "response": '{"intro": "Here are some recommendations.", "recommendations": [{"recommendation_number": 1, "summary": "A strong political fantasy choice.", "match_confidence": "high"}]}',
            "retrieved_documents": [
                Document(
                    page_content="Desc A",
                    metadata={
                        "title": "Book A",
                        "authors": "Alice",
                        "thumbnail": "https://example.com/a.jpg",
                    },
                )
            ],
        }

        mock_set_retrieving_strategy.return_value = retriever
        mock_build_chain.return_value = chain

        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)
        result = service.answer_query("hello")

        self.assertEqual(result["response"], "Here are some recommendations.")
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0]["title"], "Book A")
        self.assertEqual(result["recommendations"][0]["author"], "Alice")
        self.assertIsNone(result["recommendations"][0]["num_pages"])
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
    def test_answer_query_keeps_empty_recommendations_from_llm(
        self,
        mock_set_retrieving_strategy,
        mock_build_chain,
    ):
        """Test that an empty LLM recommendation list is returned as-is."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        mock_set_retrieving_strategy.return_value = MagicMock(name="Retriever")
        mock_build_chain.return_value = MagicMock(
            return_value={
                "response": '{"intro": "Sorry, no suitable recommendation was found based on your request.", "recommendations": []}',
                "retrieved_documents": [
                    Document(
                        page_content="Desc A",
                        metadata={
                            "title": "Book A",
                            "authors": "Alice",
                            "thumbnail": "https://example.com/a.jpg",
                        },
                    )
                ],
            }
        )

        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)
        result = service.answer_query("hello")

        self.assertEqual(
            result["response"],
            "Sorry, no suitable recommendation was found based on your request.",
        )
        self.assertEqual(result["recommendations"], [])

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

    @patch("src.rag.rag_service.build_chain")
    @patch("src.rag.rag_service.set_retrieving_strategy")
    def test_answer_query_rejects_invalid_retrieved_documents(self, mock_set_retrieving_strategy, mock_build_chain):
        """Test that if the chain returns retrieved documents in an invalid format, a RuntimeError is raised."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        mock_set_retrieving_strategy.return_value = MagicMock(name="Retriever")
        mock_build_chain.return_value = MagicMock(
            return_value={
                "response": json.dumps({"intro": "Hello", "recommendations": []}),
                "retrieved_documents": "bad-documents",
            }
        )
        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)

        with self.assertRaisesRegex(RuntimeError, "invalid retrieved documents"):
            service.answer_query("hello")

    @patch("src.rag.rag_service.build_chain")
    @patch("src.rag.rag_service.set_retrieving_strategy")
    def test_answer_query_falls_back_when_structured_parsing_fails(
        self,
        mock_set_retrieving_strategy,
        mock_build_chain,
    ):
        """Test that if the LLM response cannot be parsed as JSON, the service falls back to using the raw response and building fallback recommendation cards."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        mock_set_retrieving_strategy.return_value = MagicMock(name="Retriever")
        mock_build_chain.return_value = MagicMock(
            return_value={
                "response": "not-json-response",
                "retrieved_documents": [
                    Document(
                        page_content="A" * 300,
                        metadata={
                            "title": "Book A",
                            "author": "Alice",
                            "thumbnail": "https://example.com/a.jpg",
                            "num_pages": 321,
                        },
                    )
                ],
            }
        )

        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)
        result = service.answer_query("hello")

        self.assertEqual(result["response"], "Here are the closest books from the catalog.")
        self.assertEqual(result["raw_response"], "not-json-response")
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0]["title"], "Book A")
        self.assertEqual(result["recommendations"][0]["author"], "Alice")
        self.assertEqual(result["recommendations"][0]["num_pages"], 321)
        self.assertTrue(result["recommendations"][0]["summary"].endswith("..."))

    @patch("src.rag.rag_service.build_chain")
    @patch("src.rag.rag_service.set_retrieving_strategy")
    def test_answer_query_keeps_original_response_when_fallback_has_no_cards(
        self,
        mock_set_retrieving_strategy,
        mock_build_chain,
    ):
        """Test that if structured parsing fails but there are no retrieved documents to build fallback cards from, the original LLM response is kept."""
        llm = MagicMock(name="ChatOpenAI")
        vectorstore = MagicMock(name="FAISS")
        mock_set_retrieving_strategy.return_value = MagicMock(name="Retriever")
        mock_build_chain.return_value = MagicMock(
            return_value={
                "response": "not-json-response",
                "retrieved_documents": [],
            }
        )

        service = RAGService(llm=llm, vectorstore=vectorstore, k=3)
        result = service.answer_query("hello")

        self.assertEqual(result["response"], "not-json-response")
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["raw_response"], "not-json-response")


if __name__ == "__main__":
    unittest.main()
