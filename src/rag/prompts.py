"""Prompt loading helpers for the RAG pipeline."""

from pathlib import Path

from jinja2 import Template
from langchain_core.prompts import ChatPromptTemplate


DEFAULT_GENERATION_PROMPT_PATH = (
    Path(__file__).resolve().parent / "templates" / "default_generation_prompt.jinja2"
)
QUERY_RECOGNITION_PROMPT_PATH = (
    Path(__file__).resolve().parent / "templates" / "query_recognition_prompt.jinja2"
)
FOLLOW_UP_BACKGROUND_PROMPT_PATH = (
    Path(__file__).resolve().parent / "templates" / "follow_up_background_prompt.jinja2"
)
FOLLOW_UP_SYNTHESIS_PROMPT_PATH = (
    Path(__file__).resolve().parent / "templates" / "follow_up_synthesis_prompt.jinja2"
)


def _load_prompt_template(path: Path) -> str:
    """
    Read a prompt template from disk.

    :param path: Path to the template file.
    :type path: Path
    :return: Template contents.
    :rtype: str
    """
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    template = path.read_text(encoding="utf-8").strip()
    if not template:
        raise ValueError(f"Prompt template file is empty: {path}")
    return template


def get_default_generation_prompt() -> ChatPromptTemplate:
    """
    Build the default generation prompt from a Jinja template file.

    :return: Prompt template to ingest in the LLM chain.
    :rtype: ChatPromptTemplate
    """
    prompt_template = _load_prompt_template(DEFAULT_GENERATION_PROMPT_PATH)
    return ChatPromptTemplate.from_template(prompt_template, template_format="jinja2")


def get_query_recognition_system_prompt(
    *,
    has_history: bool,
    recommended_books: list[str],
    recommended_books_summary: str,
    recent_messages: list[str],
) -> str:
    """
    Render the query-recognition system prompt from a Jinja template.

    :param has_history: Whether the conversation already has recommendation history.
    :type has_history: bool
    :param recommended_books: Human-readable list of recommended books.
    :type recommended_books: list[str]
    :param recommended_books_summary: Summary of the recommendation history.
    :type recommended_books_summary: str
    :return: Rendered classifier system prompt.
    :rtype: str
    """
    prompt_template = _load_prompt_template(QUERY_RECOGNITION_PROMPT_PATH)
    template = Template(prompt_template)
    return template.render(
        has_history=has_history,
        recommended_books=recommended_books,
        recommended_books_summary=recommended_books_summary or "None",
        recent_messages=recent_messages,
    ).strip()


def get_follow_up_background_system_prompt() -> str:
    """
    Render the follow-up background system prompt from a Jinja template.

    :return: Rendered background system prompt.
    :rtype: str
    """
    prompt_template = _load_prompt_template(FOLLOW_UP_BACKGROUND_PROMPT_PATH)
    template = Template(prompt_template)
    return template.render().strip()


def get_follow_up_synthesis_system_prompt(
    *,
    has_sources: bool,
) -> str:
    """
    Render the follow-up synthesis system prompt from a Jinja template.

    :param has_sources: Whether the final answer will include visible source links.
    :type has_sources: bool
    :return: Rendered synthesis system prompt.
    :rtype: str
    """
    prompt_template = _load_prompt_template(FOLLOW_UP_SYNTHESIS_PROMPT_PATH)
    template = Template(prompt_template)
    return template.render(has_sources=has_sources).strip()
