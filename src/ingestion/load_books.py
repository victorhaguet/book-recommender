"""
This module is responsible for loading the dataset for our project 
into a Pandas dataframe. It also cleans the data by removing unwanted 
columns and rows. Finally, it creates a list of LangChain documents, 
with each document corresponding to a book. The page document contains 
the descriptions, while the metadata contains all the other information 
that we wanted to keep.

Note: This code only works if you have a 'description' variable in your dataset.
"""
from typing import List

import pandas as pd
from langchain_core.documents import Document

from src.logging_utils import get_logger


logger = get_logger(__name__)

def load_books(path: str, columns_to_drop: List[str]) -> List[Document]:
    """
    Load books data from a CSV file, clean it, and convert it to a list of Document objects.
    
    :param path: path of the CSV file
    :type path: str
    :param columns_to_drop: list of column names to drop from the dataframe
    :type columns_to_drop: list[str]
    :return: list of Document objects (one document per book)
    :rtype: list[Document]
    """
    logger.info("Loading books dataset from '%s'", path)
    df: pd.DataFrame = __load_dataset(path)
    clean_df: pd.DataFrame = __clean_dataframe(df, columns_to_drop)
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


def __clean_dataframe(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """
    Clean the dataframe by dropping specified columns and removing rows with missing descriptions.

    :param df: DataFrame to clean
    :type df: pd.DataFrame
    :param columns_to_drop: list of column names to drop from the dataframe
    :type columns_to_drop: list[str]
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

    df = df.drop(columns=columns_to_drop)
    df = df[df['description'].notna()]
    logger.info(
        "Cleaned dataset: dropped %d columns, removed %d rows without description, %d rows remain",
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
        page_content = row["description"]
        metadata = {key: row[key] for key in df.columns if key != 'description'}
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    logger.debug("Converted %d dataframe rows into documents", len(docs))
    return docs
