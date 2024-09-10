
from sklearn.metrics.pairwise import cosine_similarity
from couchbase.cluster import QueryOptions
from indox.core import Document
from typing import List, Callable
import numpy as np
class Couchbase:
    def __init__(self, embedding_function: Callable[[str], np.ndarray], bucket_name: str,
                 cluster_url: str = 'couchbase://localhost', username: str = 'Administrator',
                 password: str = 'indoxmain'):
        from couchbase.cluster import Cluster
        from couchbase.auth import PasswordAuthenticator
        from couchbase.exceptions import CouchbaseException
        try:
            self.cluster = Cluster(cluster_url, authenticator=PasswordAuthenticator(username, password))
            self.bucket = self.cluster.bucket(bucket_name)
            self.collection = self.bucket.default_collection()
            self.embedding_function = embedding_function

            self.create_primary_index(bucket_name)

        except CouchbaseException as e:
            raise RuntimeError(f"Failed to initialize Couchbase connection: {e}")

    def create_primary_index(self, bucket_name: str):
        """
        Create a primary index on the bucket to enable N1QL queries if it doesn't already exist.
        """
        try:
            # Check if the primary index already exists
            check_index_query = f"SELECT * FROM system:indexes WHERE name='#primary' AND keyspace_id='{bucket_name}'"
            index_exists = list(self.cluster.query(check_index_query).rows())
            if not index_exists:
                index_query = f"CREATE PRIMARY INDEX ON `{bucket_name}`"
                self.cluster.query(index_query).execute()
                print(f"Primary index created on bucket `{bucket_name}`.")
            else:
                print(f"Primary index already exists on bucket `{bucket_name}`.")
        except Exception as e:
            print(f"Failed to create primary index: {e}")

    def add(self, docs: List[str]):
        import hashlib
        for doc in docs:
            doc_key = hashlib.md5(doc.encode('utf-8')).hexdigest()

            embedding = self.embedding_function.embed_query(doc)

            self.collection.upsert(doc_key, {
                'text': doc,
                'embedding': embedding
            })

    def similarity_search_with_score(self, query: str, k: int = 3):
        try:
            query_embedding = self.embedding_function.embed_query(query)
            n1ql_query = 'SELECT meta().id, text, embedding FROM `QA` LIMIT $k'
            result = self.cluster.query(n1ql_query, QueryOptions(named_parameters={'k': k}))

            docs_with_scores = []
            for row in result:
                doc_data = row
                if 'text' in doc_data and 'embedding' in doc_data:
                    document_text = doc_data['text']
                    document_embedding = (doc_data['embedding'])

                    score = cosine_similarity([query_embedding], [document_embedding])[0][0]

                    # Create Document object with metadata
                    document = Document(page_content=document_text, id=doc_data['id'])
                    docs_with_scores.append((document, score))

            return docs_with_scores
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")