import numpy as np
from .cluster.Clustering import perform_clustering
import pandas as pd
from typing import List
from .utils import read_config
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import os


def embedding_model():
    """
    Retrieve the embedding model based on the configuration settings.

    Returns:
    - embeddings: An instance of the appropriate embeddings model as specified in the configuration.

    Raises:
    - KeyError: If the `embedding_model` key is missing in the configuration.
    - ValueError: If the specified embedding model name isn't recognized.

    Notes:
    - The function relies on the `read_config` method to determine which model to use.
    - The function currently supports 'openai' and 'sbert' as embedding models.
    """
    # Load configuration settings
    config = read_config()

    # Verify the embedding model type from the configuration
    embedding_model_name = config['embedding_model'].lower()

    if embedding_model_name == 'openai':
        model = "text-embedding-3-small"
        embeddings = OpenAIEmbeddings(model=model, openai_api_key=os.environ["OPENAI_API_KEY"])
        return embeddings
    elif embedding_model_name == 'sbert':
        embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-cos-v1")
        return embeddings
    else:
        raise ValueError(f"Unrecognized embedding model specified in the configuration: {embedding_model_name}")


def embed(texts: List[str], embeddings) -> np.ndarray:
    """
    Generate embeddings for a list of text documents using a provided embeddings object.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    text_embeddings = embeddings.embed_documents(texts)

    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_cluster_texts(texts: List[str], embeddings) -> pd.DataFrame:
    config = read_config()
    """
    Embeds and clusters a list of texts using a provided embeddings object, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = embed(
        texts, embeddings
    )  # Generate embeddings using the provided embeddings object
    cluster_labels = perform_clustering(
        text_embeddings_np,
        config["clustering"]["dim"],
        config["clustering"]["threshold"],
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df
