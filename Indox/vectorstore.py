from abc import ABC, abstractmethod
from typing import Iterable
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.chroma import Chroma
import logging
from .utils import read_config, construct_postgres_connection_string

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

class VectorStoreBase(ABC):
    """Abstract base class defining the interface for vector-based document stores."""

    @abstractmethod
    def add_document(self, texts):
        """Add documents to the vector store.

        Args:
            texts (Iterable[str]): An iterable containing the text of the documents to be added.
        
        Raises:
            RuntimeError: If there's an issue adding documents to the vector store.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str):
        """Retrieve documents similar to the given query from the vector store.

        Args:
            query (str): The query text.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            tuple: A tuple containing lists of retrieved document contexts and their similarity scores.
        """
        pass


class PGVectorStore(VectorStoreBase):
    """A concrete implementation of VectorStoreBase using PostgreSQL for storage."""

    def __init__(self, conn_string, collection_name, embedding) -> None:
        """Initialize the PGVectorStore.

        Args:
            conn_string (str): Connection string for PostgreSQL database.
            collection_name (str): Name of the collection/table in the database.
            embedding: Embedding function for vectorization.
        """
        
        super().__init__()
        self.db = PGVector(
                embedding_function=embedding,
                collection_name=collection_name,
                connection_string=conn_string,
            )

    def add_document(self, texts: Iterable[str]):
        """Add documents to the PostgreSQL vector store.

        Args:
            texts (Iterable[str]): An iterable containing the text of the documents to be added.
        
        Raises:
            RuntimeError: If there's an issue adding documents to the vector store.
        """

        try:
            self.db.add_texts(texts)
            logging.info(f"document added successfuly to the vector store")
        except:
            raise RuntimeError("Can't add document to the vector store")
        
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve documents similar to the given query from the PostgreSQL vector store.

        Args:
            query (str): The query text.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            tuple: A tuple containing lists of retrieved document contexts and their similarity scores.
        """

        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores

class ChromaVectorStore(VectorStoreBase):
    """A concrete implementation of VectorStoreBase using Chroma for storage."""
    def __init__(self, collection_name, embedding) -> None:
        """Initialize the ChromaVectorStore.

        Args:
            collection_name (str): Name of the collection/table in Chroma.
            embedding: Embedding function for vectorization.
        """

        super().__init__()
        self.db = Chroma(collection_name=collection_name,
                          embedding_function=embedding)
    
    def add_document(self, texts: Iterable[str]):
        """Add documents to the Chroma vector store.

        Args:
            texts (Iterable[str]): An iterable containing the text of the documents to be added.
        
        Raises:
            RuntimeError: If there's an issue adding documents to the vector store.
        """

        try:
            self.db.add_texts(texts)
            logging.info(f"document added successfuly to the vector store")
        except:
            raise RuntimeError("Can't add document to the vector store")
    
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve documents similar to the given query from the Chroma vector store.

        Args:
            query (str): The query text.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            tuple: A tuple containing lists of retrieved document contexts and their similarity scores.
        """

        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores

        
def get_vector_store(collection_name, embeddings):
    """Factory function to get an instance of VectorStoreBase based on configuration.

    Args:
        collection_name (str): Name of the collection/table in the database.
        embeddings: Embedding function for vectorization.

    Returns:
        VectorStoreBase: An instance of VectorStoreBase.
    """

    config = read_config()
    if config['vector_store'] == 'pgvector':
        conn_string = construct_postgres_connection_string()
        db = PGVectorStore(conn_string=conn_string,
                                collection_name=collection_name,
                                embedding=embeddings)
        return db
    elif config['vector_store'] == 'chroma':
        db = ChromaVectorStore(collection_name=collection_name,
                                embedding=embeddings)
        return db