""" This module is responsible for selecting the retrieval strategy. 
As the objective of this repository is to develop a basic RAG framework, 
only one strategy will be used for the retrieval process (i.e. k-retrieving).

Users can select the number of retrieved chunks using their configuration 
files (.env). They need to set the expected value to the RETRIEVER_K variable.
By default, k will be set to 5.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

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
    if vectorstore is None:
        raise ValueError("vectorstore must not be None")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    try:
        return vectorstore.as_retriever(k = k)
    except Exception as e:
        raise RuntimeError("Failed to create retriever from vectorstore") from e