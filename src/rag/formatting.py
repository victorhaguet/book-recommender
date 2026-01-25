""" This module is responsible for formatting the retrieved documents.
It provides a function to format the retrieved documents into a string
that can be used in a prompt template. 

By defautlt, one simple formatting strategy is implemented. 
Other strategies can be added later on if needed. 
"""

from typing import List
from langchain_core.documents import Document

def build_default_retrieved_books_format(books:List[Document])-> str:
    """
    Creation of a default formatting for the retrieved books.
    
    :param books: List of retrieved books.
    :type books: List[Document]
    :return: Formatted string of retrieved books.
    :rtype: str
    """

    if not books:
        raise ValueError("books must not be empty")

    formatted_books: str = ""

    for i, book in enumerate(books, start=1):
        if not isinstance(book, Document): 
            raise ValueError("Each item in the books list must be a Document instance")

        metadata_str = ", ".join(f"{key}: {value}\n" for key, value in book.metadata.items())
        page_content = book.page_content

        formatted_books += (f"Recommendation {i}:\n"
                            f"Description: {page_content}\n\n"
                            f"{metadata_str}"
                            f"\n--------------------\n")

    return formatted_books
