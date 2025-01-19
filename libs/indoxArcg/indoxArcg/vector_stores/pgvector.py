from indoxArcg.core import Document
from loguru import logger
import sys

from indoxArcg.core.vectorstore import VectorStore

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class PGVector(VectorStore):
    """
    A concrete implementation of VectorStoreBase using PostgreSQL for storage.

    Attributes:
        db (PGVector): The PostgreSQL vector store.
    """

    def __init__(self, host, port, dbname, user, password, collection_name, embedding):
        """
        Initializes the PGVectorStore.

        Args:
            host (str): The host of the PostgreSQL database.
            port (int): The port of the PostgreSQL database.
            dbname (str): The name of the database.
            user (str): The user for the PostgreSQL database.
            password (str): The password for the PostgreSQL database.
            collection_name (str): The name of the collection in the database.
            embedding (Embedding): The embedding to be used.
        """
        from langchain_community.vectorstores.pgvector import PGVector
        from .pgvector_setup import PGVectorSetup as PGVector
        from .pgvector_setup import DistanceStrategy as PGDistancesTRATEGY

        self.embeddings = embedding
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.db = PGVector(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_string=self._build_conn_string(),
            distance_strategy=PGDistancesTRATEGY.COSINE,
        )

    def _build_conn_string(self):
        """
        Builds the connection string from the provided components.

        Returns:
            str: The connection string.
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"

    def add(self, docs):
        """
        Adds documents to the PostgreSQL vector store.

        Args:
            docs: The documents to be added to the vector store.
        """
        try:
            if isinstance(docs[0], Document):
                self.db.add_documents(documents=docs)
            else:
                self.db.add_texts(texts=docs)
            logger.info("Document added successfully to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def _similarity_search(self, query: str, k: int = 5):
        """
        Retrieves documents similar to the given query from the PostgreSQL vector store.

        Args:
            query (str): The query to retrieve similar documents.
            k (int, optional): The number of top similar documents to retrieve. Defaults to 5.


        """
        # retrieved = self.db.similarity_search(query, k=top_k)
        # context = [d[0].page_content for d in retrieved]
        # scores = [d[1] for d in retrieved]
        # return context, scores
        embedding = self.embeddings.embed_query(text=query)
        return self.db.similarity_search_by_vector(embedding=embedding, k=k)
