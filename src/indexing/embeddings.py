"""
This module is responsible for loading the embeddings model 
that will be used for creating the vector store and granting 
access to it. 

Please note that two different implementations have been set up:
    - OpenAI API service or OpenAI-compatible API services (OpenAIEmbeddings).
    - Hugging Face local embeddings (HuggingFaceEmbeddings).

To select the strategy you wish to use, set the .env variable 
'strategy' to 'openai' or 'hf'. Then, set all the variables 
needed to run your model properly. 

Warning : If you are planning to use and Hugging face model,
please make sure you have enough space to load it locally. 
"""

from typing import Optional

from pydantic import SecretStr
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.logging_utils import get_logger


logger = get_logger(__name__)

def get_embeddings(
        strategy: str,
        model: str,
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    )->Embeddings:
    """
    Initialise the embedding model according to the strategy chosen by the user. 
    
    :return: Embedding model that will be use. 
    :rtype: Embeddings
    """
    # Normalise strategy
    strategy = strategy.lower().strip()
    logger.info("Initializing embeddings with strategy '%s' and model '%s'", strategy, model)

    # Check if the model variable is instantiate and is a string
    if not model or not isinstance(model, str):
        raise ValueError("Model variable must be a non-empty string")

    if strategy == "openai":
        # Check if the api_key is instantiate
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Missing API key for OpenAI embeddings strategy")

        # Instantiate OpenAIEmbeddings
        logger.info("Using OpenAI-compatible embeddings backend")
        return OpenAIEmbeddings(
            model = model,
            api_key= SecretStr(api_key),
            base_url= base_url
        )

    elif strategy == "hf":
        logger.info("Using Hugging Face local embeddings backend")
        return HuggingFaceEmbeddings(
            model_name = model
        )

    else:
        raise ValueError(f"Unsupported embeddings strategy: '{strategy}'. Strategy must be 'openai' or 'hf'")
