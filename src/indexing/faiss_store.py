"""
This module is responsible for managing Faiss vector stores.

It provides three functionalities:
    - Creation of a vector store
    - Saving a vector store. 
    - Loading a vector store.

This repository only allows the use of FAISS. 
Other databases like Chroma and Weaviate aren't accessible. 
"""
from typing import List

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

def create_faiss_db(embeddings: Embeddings)-> FAISS:
    """
    Creation of an empty FAISS vectorstore
    
    :param embeddings: Embedding model that will be use to create the vectors
    :type embeddings: Embeddings
    :return: Empty FAISS database
    :rtype: FAISS
    """
    # Get vector length
    index = faiss.IndexFlatL2(len(embeddings.embed_query("Hello books lovers !")))

    # Creation of the vectorstore
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store

def add_documents(documents: List[Document], vectorstore: FAISS)-> None:
    """
    Add documents to a FAISS database.
    
    :param documents: Documents to encode.
    :type documents: List[Document]
    :param vectorstore: Store to fill. 
    :type vectorstore: FAISS
    """
    vectorstore.add_documents(
        documents=documents
    )

def save_store(vectorstore: FAISS, path: str, index_name: str)-> None:
    """
    Save a existing database locally. 
    
    :param vectorstore: Vectorstore to save
    :type vectorstore: FAISS
    :param path: Location where the vectorstore should be save
    :type path: str
    :param index_name: Name of the vectorstore
    :type index_name: str
    """
    vectorstore.save_local(
        folder_path=path,
        index_name=index_name
    )

def load_store(embeddings: Embeddings, path: str, index_name:str)->FAISS:
    """
    Load a local vectorstore
    
    :param embeddings: Embedding model used to create the vectostore to load.
    :type embeddings: Embeddings
    :param path: Path of the vectostore.
    :type path: str
    :param index_name: Name of the vectorstore.
    :type index_name: str
    :return: Vectorstore loaded. 
    :rtype: FAISS
    """
    vectorstore = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        index_name=index_name
    )

    return vectorstore
