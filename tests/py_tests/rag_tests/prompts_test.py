"""Tests for prompt helpers."""
import unittest
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from src.rag.prompts import DEFAULT_GENERATION_PROMPT_PATH, get_default_generation_prompt


class TestPrompts(unittest.TestCase):
    def test_default_prompt_template_file_exists(self):
        """Test that the Jinja prompt file exists."""
        self.assertIsInstance(DEFAULT_GENERATION_PROMPT_PATH, Path)
        self.assertTrue(DEFAULT_GENERATION_PROMPT_PATH.is_file())

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


if __name__ == "__main__":
    unittest.main()
