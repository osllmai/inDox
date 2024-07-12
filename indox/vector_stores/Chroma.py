# from langchain_core.documents import Document
from indox.core import Document
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")

class ChromaVectorStore:
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
        self.embeddings = embedding
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
            logger.info("Document added successfully to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
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
            logger.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Can't retrieve documents from the vector store: {e}")
