from typing import List, Dict, Tuple
from indox.core import Document
import os

class Deeplake:
    """
    A class to interface with a DeepLake-based vector store for storing and searching text data using embeddings.

    Attributes:
    - collection_name (str): The name of the collection to store in the vector store.
    - embedding_function (callable): A function that takes text input and returns an embedding vector.

    Methods:
    - __init__(collection_name, embedding_function):
        Initializes the Deeplake instance with a specified collection name and embedding function.

    - add(docs, metadata=None):
        Adds a list of documents and optional metadata to the vector store.

        Parameters:
        - docs (List[str]): A list of text documents to be added to the vector store.
        - metadata (List[Dict], optional): A list of dictionaries containing metadata for each document.
          If not provided, default metadata with the source file path is generated.

        Raises:
        - ValueError: If `docs` is not a list of strings or if `metadata` is not a list of dictionaries.

    - similarity_search_with_score(query, k=5):
        Performs a similarity search on the vector store with the provided query and returns the most similar documents along with their scores.

        Parameters:
        - query (str): The query text for which similar documents are to be found.
        - k (int, optional): The number of top results to return. Default is 5.

        Returns:
        - List[Tuple[Document, float]]: A list of tuples where each tuple contains a `Document` object and its corresponding similarity score.

    Raises:
    - FileNotFoundError: If the specified collection path does not exist.
    - RuntimeError: If there is an issue with the vector store operation.
    """
    def __init__(self, collection_name: str, embedding_function):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        from deeplake.core.vectorstore import VectorStore

        # Initialize the VectorStore with the specified collection name
        self.vector_store = VectorStore(
            path=f"/content/vector_store/{self.collection_name}"
        )

    def add(self, docs: List[str], metadata: List[Dict] = None):
        if not isinstance(docs, list) or not all(isinstance(text, str) for text in docs):
            raise ValueError("texts must be a list of strings")

        if metadata is None:
            # Generate default metadata if none is provided
            metadata = [{"source": os.path.abspath(doc)} for doc in docs]
        elif not isinstance(metadata, list) or not all(isinstance(meta, dict) for meta in metadata):
            raise ValueError("metadata must be a list of dictionaries")

        documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(docs, metadata)]

        self.vector_store.add(
            text=docs,
            embedding_function=self.embedding_function,
            embedding_data=docs,
            metadata=metadata
        )

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        # Perform the search
        results = self.vector_store.search(
            embedding_data=query,
            embedding_function=self.embedding_function
        )

        scores = results['score']
        ids = results['id']
        metadata = results['metadata']
        texts = results['text']

        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadata)]

        return [(doc, score) for doc, score in zip(documents, scores)]

