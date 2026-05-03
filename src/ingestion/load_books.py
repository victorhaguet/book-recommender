"""
This module loads the books dataset, removes unusable rows, and converts the
remaining entries into LangChain documents.

The indexed text is built from descriptive metadata plus the original
description so retrieval has more context than the raw blurb alone.
"""
import re
from typing import List

import pandas as pd
from langchain_core.documents import Document

from src.logging_utils import get_logger


logger = get_logger(__name__)

MIN_DESCRIPTION_WORDS = 10
UNKNOWN_VALUE = "Unknown"

def load_books(
    path: str,
    columns_to_drop: List[str],
    min_description_words: int = MIN_DESCRIPTION_WORDS,
) -> List[Document]:
    """
    Load books data from a CSV file, clean it, and convert it to a list of Document objects.
    
    :param path: path of the CSV file
    :type path: str
    :param columns_to_drop: list of column names to drop from the dataframe
    :type columns_to_drop: list[str]
    :param min_description_words: minimum word count required for a description
        to be kept for indexing
    :type min_description_words: int
    :return: list of Document objects (one document per book)
    :rtype: list[Document]
    """
    if not isinstance(min_description_words, int) or min_description_words < 0:
        raise ValueError("min_description_words must be a non-negative integer")

    logger.info("Loading books dataset from '%s'", path)
    df: pd.DataFrame = __load_dataset(path)
    clean_df: pd.DataFrame = __clean_dataframe(df, columns_to_drop, min_description_words)
    docs: List[Document] = __dataframe_to_documents(clean_df)
    logger.info("Loaded %d books into LangChain documents", len(docs))
    return docs

def __load_dataset(path: str) -> pd.DataFrame:
    """
    Load books data from a CSV file.

    :param path: path of the CSV file
    :type path: str
    :return: DataFrame containing the books data
    :rtype: DataFrame
    """
    try:
        df = pd.read_csv(path, delimiter=',')
        logger.info("Read dataset '%s' with %d rows and %d columns", path, len(df), len(df.columns))
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file not found: {path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse dataset file: {path}") from e


def __count_words(text: str) -> int:
    """
    Count word-like tokens in a text.

    :param text: Text to inspect.
    :type text: str
    :return: Number of word-like tokens.
    :rtype: int
    """
    return len(re.findall(r"\b\w+\b", text))


def __is_missing_value(value: object) -> bool:
    """
    Return whether a value should be treated as missing metadata.

    :param value: Value to inspect.
    :type value: object
    :return: True when the value is missing.
    :rtype: bool
    """
    return bool(pd.isna(value))


def __normalize_metadata_value(value: object) -> object:
    """
    Convert missing metadata values to a stable placeholder.

    :param value: Value to normalize.
    :type value: object
    :return: Original value or the unknown placeholder.
    :rtype: object
    """
    if __is_missing_value(value):
        return UNKNOWN_VALUE
    return value


def __clean_dataframe(
    df: pd.DataFrame,
    columns_to_drop: list[str],
    min_description_words: int,
) -> pd.DataFrame:
    """
    Clean the dataframe by dropping specified columns and removing rows with
    missing, blank, or too-short descriptions.

    :param df: DataFrame to clean
    :type df: pd.DataFrame
    :param columns_to_drop: list of column names to drop from the dataframe
    :type columns_to_drop: list[str]
    :param min_description_words: minimum word count required for a description
    :type min_description_words: int
    :return: cleaned DataFrame
    :rtype: DataFrame
    """
    initial_rows = len(df)

    # Make sure that all the columns that should be removed are in the dataframe
    missing_cols: List[str] = [c for c in columns_to_drop if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Requested columns to drop are missing from dataset: {missing_cols}. Please make sure to only select existing columns")

    # Make sure that description is not part of the columns to drop
    has_description: bool = "description" in columns_to_drop
    if has_description:
        raise ValueError("Attempted to drop mandatory 'description' column")

    df = df.drop(columns=columns_to_drop).copy()
    df = df[df["description"].notna()].copy()
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"] != ""].copy()
    df = df[
        df["description"].map(__count_words) >= min_description_words
    ].copy()
    logger.info(
        "Cleaned dataset: dropped %d columns, removed %d rows with missing, blank, or too-short descriptions, %d rows remain",
        len(columns_to_drop),
        initial_rows - len(df),
        len(df),
    )
    return df

def __dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert a DataFrame to a list of Document objects.

    :param df: DataFrame to convert
    :type df: pd.DataFrame
    :return: list of Document objects
    :rtype: list[Document]
    """
    docs = []
    for index, row in df.iterrows():
        description = row["description"]
        metadata = {
            key: __normalize_metadata_value(row[key])
            for key in df.columns
            if key != "description"
        }
        metadata["description"] = description
        page_content = __build_page_content(metadata, description)
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    logger.debug("Converted %d dataframe rows into documents", len(docs))
    return docs


def __build_page_content(metadata: dict[str, object], description: str) -> str:
    """
    Build the indexed text for a book from salient metadata and the description.

    :param metadata: Book metadata.
    :type metadata: dict[str, object]
    :param description: Cleaned raw description.
    :type description: str
    :return: Text to embed and index.
    :rtype: str
    """
    title = str(metadata.get("title", "")).strip()
    authors = str(metadata.get("authors", metadata.get("author", ""))).strip()
    genre = str(metadata.get("genre", metadata.get("categories", ""))).strip()

    lines = []
    if title:
        lines.append(f"Title: {title}")
    if authors:
        lines.append(f"Author: {authors}")
    if genre:
        lines.append(f"Genre: {genre}")
    lines.append(f"Description: {description}")
    return "\n".join(lines)
