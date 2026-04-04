""" This module is responsible for formatting the retrieved documents.
It provides a function to format the retrieved documents into a string
that can be used in a prompt template. 

By defautlt, one simple formatting strategy is implemented. 
Other strategies can be added later on if needed. 
"""

from typing import List
from langchain_core.documents import Document

from src.logging_utils import get_logger


logger = get_logger(__name__)

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
    formatted_books: str = ""

    for i, book in enumerate(books, start=1):
        if not isinstance(book, Document): 
            raise ValueError(f"Retrieved item at position {i} is not a Document")

        metadata_str = ", ".join(f"{key}: {value}\n" for key, value in book.metadata.items())
        page_content = book.page_content

        formatted_books += (f"Recommendation {i}:\n"
                            f"Description: {page_content}\n\n"
                            f"{metadata_str}"
                            f"\n--------------------\n")

    logger.debug("Formatted retrieved books into %d characters", len(formatted_books))
    return formatted_books
