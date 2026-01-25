""" This module is responsible for managing the RAG service.
It defines a RAGSerbice class that intergrates all the components 
needed to perform RAG operations, including retrieval, creation 
of prompts, formatting of retrieved documents, and generation.
"""

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

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer for the given query using the RAG chain. 
        
        :param query: User query
        :type query: str
        :return: Result containing the answer and retrieved documents
        :rtype: Dict[str, Any]
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")
        try:
            result=self.chain(query)
        except Exception as e:
            raise RuntimeError("Failed to get answer from RAG chain") from e
        return result
