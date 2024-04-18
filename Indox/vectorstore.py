from abc import ABC, abstractmethod
from typing import Iterable
from langchain_community.vectorstores.pgvector import PGVector
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

class VectorStoreBase(ABC):

    @abstractmethod
    def add_document(self, texts):
        pass

    @abstractmethod
    def retrieve(self, query: str):
        pass


class PGVectorStore(VectorStoreBase):
    def __init__(self, conn_string, collection_name, embedding) -> None:
        super().__init__()
        self.db = PGVector(
                embedding_function=embedding,
                collection_name=collection_name,
                connection_string=conn_string,
            )

    def add_document(self, texts: Iterable[str]):
        try:
            self.db.add_texts(texts)
            logging.info(f"document added successfuly to the vector store")
        except:
            raise RuntimeError("Can't add document to the vector store")
        
    def retrieve(self, query: str, top_k: int = 5):
        retrieved = self.db.similarity_search_with_score(query, k=top_k)
        context = [d[0].page_content for d in retrieved]
        scores = [d[1] for d in retrieved]
        return context, scores
        