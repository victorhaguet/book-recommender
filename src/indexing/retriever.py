""" This module is responsible for selecting the retrieval strategy. 
As the objective of this repository is to develop a basic RAG framework, 
only one strategy will be used for the retrieval process (i.e. k-retrieving).

Users can select the number of retrieved chunks using their configuration 
files (.env). They need to set the expected value to the RETRIEVER_K variable.
By default, k will be set to 5.
"""
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

def set_retrieving_strategy(vectorstore: FAISS, k: int = 5)-> BaseRetriever:
    """
    Create a Retriever from a VectorStore that uses the K retrieval strategy.
    
    :param vectorstore: Original vectorstore
    :type vectorstore: FAISS
    :param k: Number of chunks that must be retrieved per query
    :type k: int
    :return: Retriever
    :rtype: BaseRetriever
    """
    retriever = vectorstore.as_retriever(k = k)
    return retriever

def call_retriever(retriever: BaseRetriever, query: str)-> List[Document]:
    """
    Call a retriever
    
    :param retriever: retriever to call.
    :type retriever: BaseRetriever
    :param query: User's query.
    :type query: str
    :return: retrieved chunks.
    :rtype: List[Document]
    """
    return retriever.invoke(input=query)
