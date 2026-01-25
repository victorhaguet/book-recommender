"""Tests for prompt helpers."""
import unittest

from langchain_core.prompts import ChatPromptTemplate

from src.rag.prompts import get_default_generation_prompt


class TestPrompts(unittest.TestCase):
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
