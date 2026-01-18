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
import os
from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

STRATEGY=os.getenv("STRATEGY")

load_dotenv()

def get_embeddings()->Embeddings:
    """
    Initialise the embedding model according to the strategy chosen by the user. 
    
    :return: Embedding model that will be use. 
    :rtype: Embeddings
    """

    if STRATEGY == "openai":
        # Get .env variables
        model = os.getenv("MODEL")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")

        # Prepare kwargs for OpenAIEmbeddings
        kwargs = {
            "model":model,
            "api_key":api_key
        }

        # Add base_url if it exists
        if base_url:
            kwargs["base_url"] = base_url


        return OpenAIEmbeddings(model_kwargs=kwargs)

    elif STRATEGY == "hf":
        # Get .env variables
        model = os.getenv("MODEL")

        return HuggingFaceEmbeddings(
            model_name = model
        )

    else:
        raise ValueError("Please set the strategy variable to 'openai' or 'hf' depending on the type of embedding model you would like to use.")

