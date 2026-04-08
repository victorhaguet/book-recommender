""" This module is responsible for managing the RAG service.
It defines a RAGSerbice class that intergrates all the components 
needed to perform RAG operations, including retrieval, creation 
of prompts, formatting of retrieved documents, and generation.
"""

import json
from typing import Dict, Any, Callable, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from src.indexing.retriever import set_retrieving_strategy
from src.rag.prompts import get_default_generation_prompt
from src.rag.formatting import build_default_retrieved_books_format
from src.rag.chain import build_chain
from src.logging_utils import get_logger, summarize_text


logger = get_logger(__name__)


def _stringify(value: object, fallback: str = "N/A") -> str:
    """
    Normalize values into non-empty strings.

    :param value: Value to normalize.
    :type value: object
    :param fallback: Fallback string.
    :type fallback: str
    :return: Normalized string.
    :rtype: str
    """
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _extract_json_payload(raw_response: str) -> Dict[str, Any]:
    """
    Parse the JSON payload returned by the LLM.

    :param raw_response: Raw LLM output.
    :type raw_response: str
    :return: Parsed JSON object.
    :rtype: Dict[str, Any]
    """
    # Check if the response is wrapped in code block markdown and extract the content if so (security check)
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _build_recommendation_cards(
    payload: Dict[str, Any],
    retrieved_documents: List[Document],
) -> List[Dict[str, Any]]:
    """
    Build presentation cards from parsed LLM output and retrieved docs.

    :param payload: Parsed LLM JSON output.
    :type payload: Dict[str, Any]
    :param retrieved_documents: Retrieved documents in rank order.
    :type retrieved_documents: List[Document]
    :return: Structured recommendation cards.
    :rtype: List[Dict[str, Any]]
    """
    raw_recommendations = payload.get("recommendations", [])
    if not isinstance(raw_recommendations, list):
        raise ValueError("recommendations must be a list")

    cards: list[Dict[str, Any]] = []

    # For each book retrieved, 
    for item in raw_recommendations:
        if not isinstance(item, dict):
            continue

        # Get its rank in the list
        recommendation_number = item.get("recommendation_number")
        if not isinstance(recommendation_number, int):
            continue
        index = recommendation_number - 1
        if index < 0 or index >= len(retrieved_documents):
            continue

        # Get its metadata and the LLM summary
        document = retrieved_documents[index]
        metadata = document.metadata
        title = _stringify(metadata.get("title"))
        author = _stringify(metadata.get("authors", metadata.get("author")))
        thumbnail = _stringify(metadata.get("thumbnail"), fallback="")
        summary = _stringify(item.get("summary"))

        # Format the card and add it to the list of cards
        cards.append(
            {
                "title": title,
                "author": author,
                "summary": summary,
                "thumbnail": thumbnail or None,
                "num_pages": metadata.get("num_pages"),
            }
        )

    return cards


def _build_fallback_recommendation_cards(
    retrieved_documents: List[Document],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build recommendation cards directly from retrieved documents. This is used as a fallback when the LLM does not return structured recommendations.

    :param retrieved_documents: Retrieved documents in rank order.
    :type retrieved_documents: List[Document]
    :param limit: Maximum number of fallback cards.
    :type limit: int
    :return: Structured fallback cards.
    :rtype: List[Dict[str, Any]]
    """
    cards: list[Dict[str, Any]] = []
    for document in retrieved_documents[:limit]:
        metadata = document.metadata
        title = _stringify(metadata.get("title"))
        author = _stringify(metadata.get("authors", metadata.get("author")))
        thumbnail = _stringify(metadata.get("thumbnail"), fallback="")
        description = _stringify(document.page_content)

        summary = description
        if len(summary) > 280:
            summary = summary[:277].rstrip() + "..."

        cards.append(
            {
                "title": title,
                "author": author,
                "summary": summary,
                "thumbnail": thumbnail or None,
                "num_pages": metadata.get("num_pages"),
            }
        )

    return cards


class RAGService:
    """
    Class to handle RAG operations including retrieval, formatting,
    and response generation using a language model.
    """
    def __init__(
            self,
            llm: ChatOpenAI,
            vectorstore: FAISS,
            k: int = 5
    )-> None:

        if llm is None:
            raise ValueError("llm must not be None")
        if vectorstore is None:
            raise ValueError("vectorstore must not be None")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        logger.info("Initializing RAG service with retriever k=%d", k)

        # Set the LLM and the retriever
        self.llm: ChatOpenAI = llm
        self.retriever: BaseRetriever = set_retrieving_strategy(vectorstore=vectorstore, k=k)

        # Set the prompt and formatting function
        self.prompt: ChatPromptTemplate = get_default_generation_prompt()
        self.format: Callable[[List[Document]], str] = build_default_retrieved_books_format

        # Build the RAG chain
        self.chain: Callable[[str], Dict[str, Any]] = build_chain(
            retriever=self.retriever,
            llm=self.llm,
            prompt=self.prompt,
            format_docs=self.format
        )
        logger.info("RAG service initialized successfully")

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer for the given query using the RAG chain. 
        
        :param query: User query
        :type query: str
        :return: Result containing the answer and retrieved documents
        :rtype: Dict[str, Any]
        """
        if not query or not isinstance(query, str):
            raise ValueError("Received invalid query: query must be a non-empty string")
        try:
            logger.info("Answering user query: '%s'", summarize_text(query))
            result = self.chain(query)
        except Exception as e:
            raise RuntimeError("RAG service failed to answer query") from e

        # Post-process the result to extract recommendations and handle formatting fallbacks
        raw_response = _stringify(result.get("response"), fallback="")
        retrieved_documents = result.get("retrieved_documents", [])
        if not isinstance(retrieved_documents, list):
            raise RuntimeError("RAG service returned invalid retrieved documents")

        try:
            # Extract the structured payload from the LLM response
            payload = _extract_json_payload(raw_response)
            intro = _stringify(payload.get("intro"), fallback="Here are some recommendations.")
            recommendations = _build_recommendation_cards(payload, retrieved_documents)
            result["response"] = intro
            result["recommendations"] = recommendations
            result["raw_response"] = raw_response
        except Exception:
            logger.warning("Falling back to raw LLM response because structured parsing failed")
            result["recommendations"] = _build_fallback_recommendation_cards(retrieved_documents)
            if result["recommendations"]:
                result["response"] = "Here are the closest books from the catalog."
            result["raw_response"] = raw_response

        logger.info("Query answered successfully")
        return result
