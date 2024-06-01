import logging
from langchain_core.documents import Document

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class FAISSVectorStore:
    """
    A concrete implementation of VectorStoreBase using FAISS for storage.

    Attributes:
        db (FAISS): The FAISS vector store.
    """

    def __init__(self, embedding):
        """
        Initializes the FAISSVectorStore.

        Args:
            embedding (Embedding): The embedding to be used.
        """
        from langchain_community.vectorstores.utils import DistanceStrategy
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores.faiss import FAISS
        import faiss
        self.embeddings = embedding
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
