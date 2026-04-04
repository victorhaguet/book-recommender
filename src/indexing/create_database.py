""" This file contains the function that must be 
called to create and save a database from scratch. 
Please note that you only need to call this function once. 
Instead, simply call the load function from the faiss_store.py file. 
"""
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from src.ingestion.load_books import load_books
from src.indexing.faiss_store import create_faiss_db, add_documents, save_store
from src.logging_utils import get_logger


logger = get_logger(__name__)


def create_database(
        data_path: str, 
        columns_to_drop: List[str], 
        embeddings: Embeddings,
        index_path: str,
        index_name: str
    ) -> FAISS:
    """
    Create, fill and save a VectoStore from CSV data 
    (this will only work with the Books data format).
    
    :param data_path: Path where he CSV file can be found
    :type data_path: str
    :param columns_to_drop: Columns to ignore during the creation of the database
    :type columns_to_drop: List[str]
    :param embeddings: Embedding model to use for encoding
    :type embeddings: OpenAIEmbeddings
    :param index_path: Path where the index should be stored
    :type index_path: str
    :param index_name: Name of the stored files
    :type index_name: str
    :return: Vectorstore in FAISS format
    :rtype: FAISS
    """
    logger.info(
        "Creating FAISS database from '%s' and saving to '%s/%s'",
        data_path,
        index_path,
        index_name,
    )
    
    # Ingest data
    books: List[Document] = load_books(
        path=data_path, 
        columns_to_drop=columns_to_drop
    )
    logger.info("Loaded %d documents for index creation", len(books))

    # Create and fill vectorstore
    vectorstore: FAISS = create_faiss_db(embeddings)
    add_documents(
        documents=books,
        vectorstore=vectorstore
    )
    logger.info("Vectorstore populated with %d documents", len(books))

    # Save store
    save_store(
        vectorstore=vectorstore,
        path=index_path,
        index_name=index_name
    )
    logger.info("FAISS database creation complete")

    return vectorstore
