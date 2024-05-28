from abc import ABC, abstractmethod
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
import logging
from .utils import read_config, construct_postgres_connection_string
from langchain_core.documents import Document

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class VectorStoreBase(ABC):
    """
    Abstract base class defining the interface for vector-based document stores.

    Methods:
        add_document: Abstract method to add documents to the vector store.
        retrieve: Abstract method to retrieve documents similar to the given query from the vector store.
    """

    @abstractmethod
    def add_document(self, docs):
        """
        Add documents to the vector store.

        Args:
            docs: The documents to be added to the vector store.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve documents similar to the given query from the vector store.

        Args:
            query (str): The query to retrieve similar documents.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.
        """
        pass


class PGVectorStore(VectorStoreBase):
    """
    A concrete implementation of VectorStoreBase using PostgreSQL for storage.

    Attributes:
        db (PGVector): The PostgreSQL vector store.
    """

    def __init__(self, conn_string, collection_name, embedding):
        """
        Initializes the PGVectorStore.

        Args:
            conn_string (str): The connection string to the PostgreSQL database.
            collection_name (str): The name of the collection in the database.
            embedding (Embedding): The embedding to be used.
        """
        from langchain_community.vectorstores.pgvector import PGVector
        from langchain_community.vectorstores.pgvector import DistanceStrategy as PGDistancesTRATEGY
        super().__init__()
        self.conn_string = conn_string
        self.db = PGVector(embedding_function=embedding, collection_name=collection_name,
                           connection_string=conn_string, distance_strategy=PGDistancesTRATEGY.COSINE)

    def add_document(self, docs):
        """
        Adds documents to the PostgreSQL vector store.

        Args:
            docs: The documents to be added to the vector store.
        """
        try:
            if isinstance(docs[0], Document):
                self.db.add_documents(documents=docs)
            elif not isinstance(docs[0], Document):
                self.db.add_texts(texts=docs)
            logging.info("Document added successfully to the vector store.")
        except Exception as e:
            logging.error(f"Failed to add document: {e}")
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieves documents similar to the given query from the PostgreSQL vector store.

        Args:
            query (str): The query to retrieve similar documents.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            Tuple[List[str], List[float]]: The context and scores of the retrieved documents.
        """
        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores


class ChromaVectorStore(VectorStoreBase):
    """
    A concrete implementation of VectorStoreBase using Chroma for storage.

    Attributes:
        db (Chroma): The Chroma vector store.
    """

    def __init__(self, collection_name, embedding):
        """
        Initializes the ChromaVectorStore.

        Args:
            collection_name (str): The name of the collection in the database.
            embedding (Embedding): The embedding to be used.
        """
        from langchain_community.vectorstores.chroma import Chroma
        super().__init__()
        self.db = Chroma(collection_name=collection_name, embedding_function=embedding)

    def add_document(self, docs):
        """
        Adds documents to the Chroma vector store.

        Args:
            docs: The documents to be added to the vector store.
        """
        try:
            if isinstance(docs[0], Document):
                self.db.add_documents(documents=docs)
            elif not isinstance(docs[0], Document):
                self.db.add_texts(texts=docs)
            logging.info("Document added successfully to the vector store.")
        except Exception as e:
            logging.error(f"Failed to add document: {e}")
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieves documents similar to the given query from the Chroma vector store.

        Args:
            query (str): The query to retrieve similar documents.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            Tuple[List[str], List[float]]: The context and scores of the retrieved documents.
        """
        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores

    def get_all_documents(self):
        """
        Retrieves all documents from the Chroma vector store.

        Returns:
            List[dict]: A list of all documents with their metadata.
        """
        try:
            all_documents = self.db.get()
            return all_documents
        except Exception as e:
            logging.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Can't retrieve documents from the vector store: {e}")


class FAISSVectorStore(VectorStoreBase):
    """
    A concrete implementation of VectorStoreBase using FAISS for storage.

    Attributes:
        db (FAISS): The FAISS vector store.
    """

    def __init__(self, embedding) -> None:
        """
        Initializes the FAISSVectorStore.

        Args:
            embedding (Embedding): The embedding to be used.
        """
        from langchain_community.vectorstores.faiss import FAISS
        import faiss
        super().__init__()

        embedding_dim = len(embedding.embed_query(""))

        index = faiss.IndexFlatL2(embedding_dim)

        docstore = InMemoryDocstore({})

        index_to_docstore_id = {}

        self.db = FAISS(
            embedding,
            index,
            docstore,
            index_to_docstore_id,
            relevance_score_fn=None,
            normalize_L2=False,
            distance_strategy=DistanceStrategy.COSINE
        )

    def add_document(self, docs):
        """
        Adds documents to the FAISS vector store.

        Args:
            docs: The documents to be added to the vector store.
        """
        try:
            if isinstance(docs[0], Document):
                self.db.add_documents(documents=docs)
            elif not isinstance(docs[0], Document):
                self.db.add_texts(texts=docs)
            logging.info("Document added successfully to the vector store.")
        except Exception as e:
            logging.error(f"Failed to add document: {e}")
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieves documents similar to the given query from the FAISS vector store.

        Args:
            query (str): The query to retrieve similar documents.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            Tuple[List[str], List[float]]: The context and scores of the retrieved documents.
        """
        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores

    def get_all_documents(self):
        """
        Retrieves all documents from the Chroma vector store.

        Returns:
            List[dict]: A list of all documents with their metadata.
        """
        try:
            all_documents = self.db.get()
            return all_documents
        except Exception as e:
            logging.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Can't retrieve documents from the vector store: {e}")


def get_vector_store(embeddings, collection_name: str):
    """
    Returns the appropriate vector store based on the configuration.

    Args:
        embeddings (Embedding): The embeddings to be used.
        collection_name (str): The name of the collection in the database.

    Returns:
        VectorStoreBase: The appropriate vector store.
    """
    config = read_config()
    if config['vector_store'].lower() == 'pgvector':
        conn_string = construct_postgres_connection_string()
        return PGVectorStore(conn_string=conn_string, collection_name=collection_name,
                             embedding=embeddings)
    elif config['vector_store'].lower() == 'chroma':
        return ChromaVectorStore(collection_name=collection_name, embedding=embeddings)
    elif config['vector_store'].lower() == 'faiss':
        db = FAISSVectorStore(embedding=embeddings)
        return db
