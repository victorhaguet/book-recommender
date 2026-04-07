"""Prompt loading helpers for the RAG pipeline."""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


DEFAULT_GENERATION_PROMPT_PATH = (
    Path(__file__).resolve().parent / "templates" / "default_generation_prompt.jinja2"
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
