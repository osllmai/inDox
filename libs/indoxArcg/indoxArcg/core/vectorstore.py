import abc
from pydantic import BaseModel


class VectorStore(BaseModel, abc.ABC):
    """
    Abstract base class for vector stores.
    """

    @abc.abstractmethod
    def add_document(self, docs):
        pass

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5):
        pass

    @abc.abstractmethod
    def get_all_documents(self):
        pass

    class Config:
        arbitrary_types_allowed = True
