from langchain_core.documents import Document
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)




class PGVectorStore:
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
        self.embeddings = embedding
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