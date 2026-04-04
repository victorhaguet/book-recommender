"""Shared logging helpers for the project."""

from __future__ import annotations

import logging
import os


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> None:
    """
    Configure application logging once if nothing configured it already.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module logger after applying the default logging configuration.
    """
    configure_logging()
    return logging.getLogger(name)


def summarize_text(value: str, limit: int = 120) -> str:
    """
    Truncate long values for safer log lines.
    """
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."
