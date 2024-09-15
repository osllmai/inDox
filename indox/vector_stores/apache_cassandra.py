import uuid
import json
import numpy as np
from typing import List, Tuple, Callable
from indox.core import Document


class ApacheCassandra:
    """
    A class to interface with an Apache Cassandra-based vector store for storing and searching text data using embeddings.

    Attributes:
    - cluster_ips (List[str]): Fixed list of IP addresses of the Cassandra nodes.
    - port (int): Fixed port number to connect to the Cassandra cluster.
    - embedding_function (Callable[[str], np.ndarray]): A function that takes text input and returns an embedding vector.
    - keyspace (str): The keyspace to use in Cassandra.

    Methods:
    - __init__(embedding_function, keyspace):
        Initializes the ApacheCassandra instance with specified embedding function and keyspace.

    - add(docs):
        Adds a list of documents to the Cassandra vector store.

        Parameters:
        - docs (List[str]): A list of text documents to be added to the vector store.

        Raises:
        - ValueError: If `docs` is not a list of strings.
        - RuntimeError: If there is an issue with the Cassandra database operation.

    - similarity_search_with_score(query, k=5):
        Performs a similarity search on the Cassandra vector store with the provided query and returns the most similar documents along with their scores.

        Parameters:
        - query (str): The query text for which similar documents are to be found.
        - k (int, optional): The number of top results to return. Default is 5.

        Returns:
        - List[Tuple[Document, float]]: A list of tuples where each tuple contains a `Document` object and its corresponding similarity score.

        Raises:
        - RuntimeError: If there is an issue with the Cassandra database operation.

    - shutdown():
        Shuts down the Cassandra cluster connection.
    """

    def __init__(self, embedding_function: Callable[[str], np.ndarray], keyspace: str):
        from cassandra.cluster import Cluster
        from cassandra.policies import DCAwareRoundRobinPolicy
        try:
            self.cluster = Cluster(
                contact_points=['127.0.0.1'],
                port=9042,
                load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='datacenter1'),
                protocol_version=4
            )
            self.session = self.cluster.connect()
            self.keyspace = keyspace
            self.embedding_function = embedding_function
            self._setup_keyspace()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Cassandra connection: {e}")

    def _setup_keyspace(self):
        try:
            self.session.execute(f"""
                CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
                WITH REPLICATION = {{ 'class' : 'SimpleStrategy', 'replication_factor' : 1 }}
            """)
            self.session.set_keyspace(self.keyspace)
            self.session.execute("DROP TABLE IF EXISTS embeddings_table")
            self.session.execute(
                """CREATE TABLE IF NOT EXISTS embeddings_table (id UUID PRIMARY KEY, embedding text, chunk_text text)""")
        except Exception as e:
            raise RuntimeError(f"Failed to set up keyspace or table: {e}")

    def add(self, docs: List[str]):
        if not isinstance(docs, list):
            raise ValueError("The 'docs' argument must be a list of strings")
        try:
            chunk_embeddings = self.embedding_function.embed_documents(docs)
            for chunk, embedding in zip(docs, chunk_embeddings):
                embedding_str = json.dumps(embedding)
                self.session.execute("""INSERT INTO embeddings_table (id, embedding, chunk_text) VALUES (%s, %s, %s)""",
                                     (uuid.uuid4(), embedding_str, chunk))
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Cassandra: {e}")

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        try:
            query_embedding = np.array(self.embedding_function.embed_query(query))
            rows = self.session.execute('SELECT id, embedding, chunk_text FROM embeddings_table')
            similarity_scores = []
            for row in rows:
                stored_embedding = np.array(json.loads(row.embedding))
                similarity = cosine_similarity(stored_embedding.reshape(1, -1), query_embedding.reshape(1, -1))[0][0]
                document = Document(page_content=row.chunk_text, id=str(row.id))
                similarity_scores.append((document, similarity))
            similarity_scores.sort(reverse=True, key=lambda x: x[1])
            return similarity_scores[:k]
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")

    def shutdown(self):
        try:
            self.cluster.shutdown()
        except Exception as e:
            raise RuntimeError(f"Failed to shutdown Cassandra cluster: {e}")
