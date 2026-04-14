"""Tests for prompt helpers."""
import tempfile
import unittest
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from src.rag.prompts import (
    DEFAULT_GENERATION_PROMPT_PATH,
    FOLLOW_UP_BACKGROUND_PROMPT_PATH,
    FOLLOW_UP_SYNTHESIS_PROMPT_PATH,
    QUERY_RECOGNITION_PROMPT_PATH,
    _load_prompt_template,
    get_default_generation_prompt,
    get_follow_up_background_system_prompt,
    get_follow_up_synthesis_system_prompt,
    get_query_recognition_system_prompt,
)


class TestPrompts(unittest.TestCase):
    def test_default_prompt_template_file_exists(self):
        """Test that the Jinja prompt file exists."""
        self.assertIsInstance(DEFAULT_GENERATION_PROMPT_PATH, Path)
        self.assertTrue(DEFAULT_GENERATION_PROMPT_PATH.is_file())
        self.assertIsInstance(QUERY_RECOGNITION_PROMPT_PATH, Path)
        self.assertTrue(QUERY_RECOGNITION_PROMPT_PATH.is_file())
        self.assertIsInstance(FOLLOW_UP_BACKGROUND_PROMPT_PATH, Path)
        self.assertTrue(FOLLOW_UP_BACKGROUND_PROMPT_PATH.is_file())
        self.assertIsInstance(FOLLOW_UP_SYNTHESIS_PROMPT_PATH, Path)
        self.assertTrue(FOLLOW_UP_SYNTHESIS_PROMPT_PATH.is_file())

    def test_get_default_generation_prompt_returns_template(self):
        """Test that default prompt is a ChatPromptTemplate."""
        prompt = get_default_generation_prompt()
        self.assertIsInstance(prompt, ChatPromptTemplate)

    def test_default_prompt_formats_retrieved_books(self):
        """Test that the default prompt can format retrieved books."""
        prompt = get_default_generation_prompt()
        messages = prompt.format_messages(retrieved_books="Book A")

        self.assertTrue(messages)
        self.assertTrue(any("Book A" in message.content for message in messages))

    def test_default_prompt_mentions_in_scope_recognition(self):
        """Test that the generation prompt assumes query recognition already happened."""
        prompt = get_default_generation_prompt()
        messages = prompt.format_messages(retrieved_books="Book A")
        combined_content = "\n".join(str(message.content) for message in messages)

        self.assertIn("already been recognized as an in-scope book recommendation request", combined_content)

    def test_load_prompt_template_raises_for_missing_file(self):
        """Test that loading a non-existent prompt template file raises a FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _load_prompt_template(Path("/tmp/definitely-missing-prompt-template.jinja2"))

    def test_load_prompt_template_raises_for_empty_file(self):
        """Test that loading an empty prompt template file raises a ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja2") as handle:
            handle.write("   \n")
            handle.flush()

            with self.assertRaises(ValueError):
                _load_prompt_template(Path(handle.name))

    def test_query_recognition_prompt_renders_context(self):
        """Test that the query-recognition prompt includes rendered history context."""
        prompt = get_query_recognition_system_prompt(
            has_history=True,
            recommended_books=["Book A by Alice", "Book B by Bob"],
            recommended_books_summary="Most recent recommended books: Book A by Alice.",
            recent_messages=["user: Who is Alice?", "assistant: Alice is the author of Book A."],
        )

        self.assertIn("follow_up", prompt)
        self.assertIn("Book A by Alice", prompt)
        self.assertIn("Most recent recommended books", prompt)
        self.assertIn("Who is Alice?", prompt)

    def test_query_recognition_prompt_mentions_no_history_rule(self):
        """Test that the no-history branch includes the stricter routing rule."""
        prompt = get_query_recognition_system_prompt(
            has_history=False,
            recommended_books=[],
            recommended_books_summary="",
            recent_messages=[],
        )

        self.assertIn("Without history", prompt)
        self.assertIn("recommendation-seeking queries", prompt)

    def test_follow_up_background_prompt_renders(self):
        """Test that the follow-up background prompt renders correctly."""
        prompt = get_follow_up_background_system_prompt()

        self.assertIn("background note", prompt)
        self.assertIn("If uncertain", prompt)

    def test_follow_up_synthesis_prompt_renders_with_sources(self):
        """Test that the follow-up synthesis prompt includes source mention when requested."""
        prompt = get_follow_up_synthesis_system_prompt(has_sources=True)

        self.assertIn("Prefer Tavily and Wikipedia evidence", prompt)
        self.assertIn("sources are listed below", prompt)

    def test_follow_up_synthesis_prompt_renders_without_sources(self):
        """Test that the follow-up synthesis prompt omits source mention when there are no sources."""
        prompt = get_follow_up_synthesis_system_prompt(has_sources=False)

        self.assertIn("Do not invent facts", prompt)
        self.assertNotIn("sources are listed below", prompt)


if __name__ == "__main__":
    unittest.main()
