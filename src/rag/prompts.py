"""This module is responsible for selecting the prompt templates
to be used in the RAG framework. 

Currently, one prompt template is defined for the generation task.
This prompt is the default one. It is really simple and will be 
used as a baseline. If it is not performant enough, other prompts
can be added later on.
"""

from langchain_core.prompts import ChatPromptTemplate

def get_default_generation_prompt() -> ChatPromptTemplate:
    """
    Build a simple default prompt template for the generation task.

    :return: Prompt template to ingest in the LLM chain. 
    :rtype: ChatPromptTemplate
    """

    prompt_template= """
    You are a library assistant who helps users find books based on their preferences.
    For each book recommendation retrieved below, follow these steps:
    1. Check if the book matches the user's preferences.
    2. If it matches, present the book to the user by :
        - Stating the book title and author.
        - Providing a brief summary of the book description and why it should interest the user. 
        - Bring up any other information of the book you have access if relevant (e.g., genre, pages, publisher, etc.).
       Be as honest as possble in your recommendation (if the book is long, tell it. If the rankings are low, tell it too, etc.).
    3. If none of the books match the user's preferences, politely inform the user that no suitable recommendations were found.

    Here are the book recommendations:
    {retrieved_books}
    """

    return ChatPromptTemplate.from_template(prompt_template)