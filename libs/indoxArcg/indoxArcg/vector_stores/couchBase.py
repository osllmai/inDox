import uuid
import json
import numpy as np
from typing import List, Tuple, Callable
from indoxArcg.core import Document


class Couchbase:

    def __init__(
        self,
        embedding_function: Callable[[str], np.ndarray],
        bucket_name: str,
        cluster_url: str = "couchbase://localhost",
        username: str = "Administrator",
        password: str = "indoxmain",
    ):
        """
        Initialize a Couchbase connection and set up the embedding function.

        Args:
            embedding_function (Callable[[str], np.ndarray]): Function to embed the documents or queries.
            bucket_name (str): Name of the Couchbase bucket.
            cluster_url (str, optional): URL of the Couchbase cluster. Defaults to 'couchbase://localhost'.
            username (str, optional): Username for authentication. Defaults to 'Administrator'.
            password (str, optional): Password for authentication. Defaults to 'indoxmain'.

        Raises:
            RuntimeError: If the connection to Couchbase cannot be established.
        """
        from couchbase.cluster import Cluster
        from couchbase.auth import PasswordAuthenticator
        from couchbase.exceptions import CouchbaseException
        from couchbase.bucket import Bucket

        try:
            self.cluster = Cluster(
                cluster_url, authenticator=PasswordAuthenticator(username, password)
            )
            self.bucket = self.cluster.bucket(bucket_name)
            self.collection = self.bucket.default_collection()
            self.embedding_function = embedding_function
        except CouchbaseException as e:
            raise RuntimeError(f"Failed to initialize Couchbase connection: {e}")

    def add(self, docs: List[str]):
        """
        Add a list of documents to Couchbase after embedding them.

        Args:
            docs (List[str]): List of document strings to be added.

        Raises:
            ValueError: If the input 'docs' is not a list.
            RuntimeError: If there is an error while adding documents to Couchbase.

        Notes:
            Each document is embedded using the provided embedding function, assigned a unique
            UUID, and stored in Couchbase. The document is stored with its embedding and original text.
        """
        from couchbase.exceptions import CouchbaseException

        if not isinstance(docs, list):
            raise ValueError("The 'docs' argument must be a list of strings")

        try:
            for doc in docs:
                embedding = self.embedding_function.embed_documents(doc)

                doc_id = str(uuid.uuid4())
                doc_data = {"id": doc_id, "embedding": embedding, "text": doc}
                self.collection.upsert(doc_id, doc_data)
        except CouchbaseException as e:
            raise RuntimeError(f"Failed to add documents to Couchbase: {e}")

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search on the embedded documents and return the results with scores.

        Args:
            query (str): The query string to search for.
            k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Tuple[Document, float]]: A list of tuples containing Document objects and
            their similarity scores. Higher scores represent closer matches.

        Raises:
            RuntimeError: If there is an error during the similarity search process.

        Notes:
            The method retrieves the documents by their IDs and computes similarity scores
            by comparing the query embedding with stored document embeddings.
        """
        from couchbase.exceptions import CouchbaseException
        from couchbase.search import SearchOptions, MatchQuery

        try:
            query_embedding = self.embedding_function.embed_query(query)

            match_query = MatchQuery(query)
            search_options = SearchOptions(limit=k)
            result = self.cluster.search_query("FTS_QA", match_query, search_options)

            similarity_scores = []
            for row in result.rows():
                doc_id = row.id

                try:
                    doc_data = self.collection.get(doc_id).content_as[dict]
                    print(
                        f"Document with ID {doc_id}: {json.dumps(doc_data, indent=2)}"
                    )

                    if "answer" in doc_data:
                        document_content = doc_data["answer"]
                    elif "question" in doc_data:
                        document_content = doc_data["question"]
                    else:
                        document_content = (
                            doc_data.get("title", "")
                            + " "
                            + doc_data.get("name", "")
                            + " "
                            + doc_data.get("address", "")
                        )

                    document = Document(page_content=document_content, id=doc_id)
                    score = row.score
                    similarity_scores.append((document, score))

                except CouchbaseException as e:
                    print(f"Failed to fetch document with ID {doc_id}: {e}")

            similarity_scores.sort(reverse=True, key=lambda x: x[1])
            return similarity_scores[:k]

        except CouchbaseException as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")
