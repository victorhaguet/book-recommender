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

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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

    # Check if the model variable is instantiate and is a string
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string")

    if strategy == "openai":
        # Check if the api_key is instantiate
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string")

        # Prepare kwargs for OpenAIEmbeddings
        kwargs = {
            "model":model,
            "api_key":api_key
        }

        # Add base_url if it exists
        if base_url:
            kwargs["base_url"] = base_url


        return OpenAIEmbeddings(model_kwargs=kwargs)

    elif strategy == "hf":

        return HuggingFaceEmbeddings(
            model_name = model
        )

    else:
        raise ValueError("strategy must be 'openai' or 'hf'")
