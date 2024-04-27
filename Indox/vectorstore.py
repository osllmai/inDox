from abc import ABC, abstractmethod
from typing import Iterable
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import faiss
import logging
from .utils import read_config, construct_postgres_connection_string

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class VectorStoreBase(ABC):
    """Abstract base class defining the interface for vector-based document stores."""

    @abstractmethod
    def add_document(self, texts):
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve documents similar to the given query from the vector store."""
        pass


class PGVectorStore(VectorStoreBase):
    """A concrete implementation using PostgreSQL for storage."""

    def __init__(self, conn_string, collection_name, embedding):
        super().__init__()
        self.db = PGVector(embedding_function=embedding, collection_name=collection_name, connection_string=conn_string)

    def add_document(self, texts: Iterable[str]):
        try:
            self.db.add_texts(texts)
            logging.info("Document added successfully to the vector store.")
        except Exception as e:
            logging.error(f"Failed to add document: {e}")
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def retrieve(self, query: str, top_k: int = 5):
        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores


class ChromaVectorStore(VectorStoreBase):
    """A concrete implementation using Chroma for storage."""

    def __init__(self, collection_name, embedding):
        super().__init__()
        self.db = Chroma(collection_name=collection_name, embedding_function=embedding)

    def add_document(self, texts: Iterable[str]):
        try:
            self.db.add_texts(texts)
            logging.info("Document added successfully to the vector store.")
        except Exception as e:
            logging.error(f"Failed to add document: {e}")
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def retrieve(self, query: str, top_k: int = 5):
        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores


class FAISSVectorStore(VectorStoreBase):
    """A concrete implementation of VectorStoreBase using FAISS for storage."""

    def __init__(self, embedding) -> None:

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
            distance_strategy="euclidean_distance"
        )

    def add_document(self, texts: Iterable[str]):
        try:
            self.db.add_texts(texts=texts)
            print("document added successfuly to the vector store")
        except:
            raise RuntimeError("Can't add document to the vector store")

    def retrieve(self, query: str, top_k: int = 5):
        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores


def get_vector_store(embeddings, collection_name: str):
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
