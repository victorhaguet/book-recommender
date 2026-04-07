"""Format retrieved documents for prompt consumption."""

from typing import List
from langchain_core.documents import Document

from src.logging_utils import get_logger


logger = get_logger(__name__)


def _stringify_metadata_value(value: object) -> str:
    """
    Normalize metadata values into prompt-friendly strings.

    :param value: Metadata value to normalize.
    :type value: object
    :return: String representation for prompt use.
    :rtype: str
    """
    if value is None:
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


def build_default_retrieved_books_format(books:List[Document])-> str:
    """
    Creation of a default formatting for the retrieved books.
    
    :param books: List of retrieved books.
    :type books: List[Document]
    :return: Formatted string of retrieved books.
    :rtype: str
    """

    if not books:
        raise ValueError("Cannot format an empty list of retrieved books")

    logger.info("Formatting %d retrieved books for the prompt", len(books))
    formatted_sections: list[str] = []

    # For each retrieved book,
    for i, book in enumerate(books, start=1):
        if not isinstance(book, Document): 
            raise ValueError(f"Retrieved item at position {i} is not a Document")

        # Get and normalize the metadata values
        metadata = {key: _stringify_metadata_value(value) for key, value in book.metadata.items()}
        title = metadata.get("title", "N/A")
        authors = metadata.get("authors", metadata.get("author", "N/A"))
        thumbnail = metadata.get("thumbnail", "N/A")
        description = _stringify_metadata_value(book.page_content)

        # Format the normalized values into a structured string
        remaining_metadata_lines = [
            f"- {key}: {value}"
            for key, value in metadata.items()
            if key not in {"title", "authors", "author", "thumbnail"}
        ]
        remaining_metadata = "\n".join(remaining_metadata_lines) if remaining_metadata_lines else "- None"

        # Add the formatted section for this book to the list of retrieved books
        formatted_sections.append(
            (
                f"Recommendation {i}\n"
                f"title: {title}\n"
                f"authors: {authors}\n"
                f"thumbnail: {thumbnail}\n"
                f"description: {description}\n"
                "other_metadata:\n"
                f"{remaining_metadata}"
            )
        )

    formatted_books = "\n\n--------------------\n\n".join(formatted_sections)
    logger.debug("Formatted retrieved books into %d characters", len(formatted_books))
    return formatted_books
