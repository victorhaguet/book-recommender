""" This module is responsible for selecting the retrieval strategy.
As the objective of this repository is to develop a basic RAG framework,
only one strategy will be used for the retrieval process (k-retrieving).

This chain can evolve in the future if the current results are not
satisfactory enough. Other retrieval strategies can be implemented
later on.
"""

from typing import Callable, Dict, Any, List, cast
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_chain(
        retriever: BaseRetriever,
        llm: ChatOpenAI,
        prompt: ChatPromptTemplate,
        format_docs: Callable[[List[Document]], str]
    )-> Callable[[str], Dict[str, Any]]:
    """
    Build a RAG chain that retrieves documents, formats them,
    and generates a response using an LLM.
    
    :param retriever: Retriever to fetch relevant documents.
    :type retriever: BaseRetriever
    :param llm: Language model to generate responses.
    :type llm: ChatOpenAI
    :param prompt: Prompt template for the LLM.
    :type prompt: ChatPromptTemplate
    :param format_docs: Function to format retrieved documents.
    :type format_docs: Callable[[List[Document]], str]
    :return: A callable that takes a query and returns a dictionary with the response and retrieved documents.
    :rtype: Callable[[str], Dict[str, Any]]
    """

    if retriever is None:
        raise ValueError("retriever must not be None")
    if llm is None:
        raise ValueError("llm must not be None")
    if prompt is None:
        raise ValueError("prompt must not be None")
    if format_docs is None or not callable(format_docs):
        raise ValueError("format_docs must be a callable")


    def chain(query: str) -> Dict[str, Any]:
        # retrieve documents
        try:
            retrieved_docs:List[Document] = retriever.invoke(input=query)
        except Exception as e:
            raise RuntimeError("Failed to retrieve documents") from e

        # format retrieved documents
        try:
            context: str = format_docs(retrieved_docs)
        except Exception as e:
            raise RuntimeError("Failed to format retrieved documents") from e

        # Build prompt input
        try:
            messages: List[BaseMessage] = prompt.format_messages(retrieved_books=context)
        except Exception as e:
            raise RuntimeError("Failed to format prompt with retrieved documents") from e

        # Call
        try:
            llm_response: AIMessage = llm.invoke(input=messages)
        except Exception as e:
            raise RuntimeError("Failed to invoke LLM with prompt messages") from e

        # Extract response
        response: str = cast(str, llm_response.content)
        if not isinstance(response, str):
            raise ValueError("LLM response content is not a string")

        return {
            "response": response,
            "retrieved_documents": retrieved_docs
        }

    return chain
