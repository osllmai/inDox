import numpy as np
from .Clustering import perform_clustering
import pandas as pd
from typing import List
from .utils import read_config
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def embedding_model():
    config = read_config()
    if config['embedding_model'] == 'openai':
        model = "text-embedding-3-small"
        # openai_api_key = config["openai_api_key"],
        embeddings = OpenAIEmbeddings(model=model, openai_api_key=OPENAI_API_KEY)
        embed_documents = embeddings.embed_documents
        return embeddings, embed_documents
    elif config['embedding_model'] == 'SBert':
        model_name = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
        embeddings = SentenceTransformer(model_name)
        embed_documents = embeddings.encode
        return embeddings, embed_documents


def embed(texts: List[str], embeddings) -> np.ndarray:
    """
    Generate embeddings for a list of text documents using a provided embeddings object.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.
    - embeddings: An object capable of generating embeddings, must have an `embed_documents` method.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    embeddings, embed_documents = embedding_model()
    text_embeddings = embed_documents(texts)

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
