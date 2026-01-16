"""
This module is responsible for loading the dataset for our project 
into a Pandas dataframe. It also cleans the data by removing unwanted 
columns and rows. Finally, it creates a list of LangChain documents, 
with each document corresponding to a book. The page document contains 
the descriptions, while the metadata contains all the other information 
that we wanted to keep.

Note: This code only works if you have a 'description' variable in your dataset.
"""

import pandas as pd
from langchain_core.documents import Document

def load_books(path: str, columns_to_drop: list[str]) -> list[Document]:
    """
    Load books data from a CSV file, clean it, and convert it to a list of Document objects.
    
    :param path: path of the CSV file
    :type path: str
    :param columns_to_drop: list of column names to drop from the dataframe
    :type columns_to_drop: list[str]
    :return: list of Document objects (one document per book)
    :rtype: list[Document]
    """
    df = __load_books(path)
    clean_df = __clean_dataframe(df, columns_to_drop)
    docs = __dataframe_to_documents(clean_df)
    return docs

def __load_books(path: str) -> pd.DataFrame:
    """
    Load books data from a CSV file.

    :param path: path of the CSV file
    :type path: str
    :return: DataFrame containing the books data
    :rtype: DataFrame
    """
    df = pd.read_csv(path, delimiter=',')
    return df

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
    df = df.drop(columns=columns_to_drop)
    df = df[df['description'].notna()]
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
    return docs