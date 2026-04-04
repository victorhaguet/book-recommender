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

from src.logging_utils import get_logger


logger = get_logger(__name__)

def create_faiss_db(embeddings: Embeddings)-> FAISS:
    """
    Creation of an empty FAISS vectorstore
    
    :param embeddings: Embedding model that will be use to create the vectors
    :type embeddings: Embeddings
    :return: Empty FAISS database
    :rtype: FAISS
    """
    if embeddings is None:
        logger.error("Cannot create FAISS database without embeddings")
        raise ValueError("embeddings must not be None")

    try:
        vec = embeddings.embed_query("Hello books lovers !")
    except Exception as e:
        raise RuntimeError("Failed to compute embedding dimension via embed_query()") from e
    
    if vec==[]:
        raise ValueError("Embedding backend returned an empty vector")

    logger.info("Creating empty FAISS index with embedding dimension %d", len(vec))
    
    index = faiss.IndexFlatL2(len(vec))

    # Creation of the vectorstore
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

def add_documents(documents: List[Document], vectorstore: FAISS)-> None:
    """
    Add documents to a FAISS database.
    
    :param documents: Documents to encode.
    :type documents: List[Document]
    :param vectorstore: Store to fill. 
    :type vectorstore: FAISS
    """

    if vectorstore is None:
        raise ValueError("Cannot add documents to a missing vectorstore")

    logger.info("Adding %d documents to FAISS vectorstore", len(documents))
    vectorstore.add_documents(documents=documents)
    logger.info("Documents added to FAISS vectorstore")

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
    if vectorstore is None:
        raise ValueError("Cannot save a missing vectorstore")

    if not path or not isinstance(path, str):
        raise ValueError("Invalid FAISS save path")

    if not index_name or not isinstance(index_name, str):
        raise ValueError("Invalid FAISS index name")

    logger.info("Saving FAISS vectorstore to '%s' with index name '%s'", path, index_name)
    vectorstore.save_local(folder_path=path, index_name=index_name)
    logger.info("FAISS vectorstore saved successfully")

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
    if embeddings is None:
        raise ValueError("Cannot load FAISS vectorstore without embeddings")

    if not path or not isinstance(path, str):
        raise ValueError("Invalid FAISS load path")

    if not index_name or not isinstance(index_name, str):
        raise ValueError("Invalid FAISS index name")

    try:
        logger.info("Loading FAISS vectorstore from '%s' with index name '%s'", path, index_name)
        store = FAISS.load_local(
            folder_path=path, 
            embeddings=embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS vectorstore loaded successfully")
        return store
    except FileNotFoundError:
        raise FileNotFoundError(f"FAISS vectorstore files were not found at '{path}' with index '{index_name}'")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FAISS store from '{path}' (index_name='{index_name}'). "
            "Embeddings must be compatible with the index."
        ) from e
