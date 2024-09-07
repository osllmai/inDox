import uuid
import json
import numpy as np
from typing import List, Tuple, Callable
from indox.core import Document


class Couchbase:

    def __init__(self, embedding_function: Callable[[str], np.ndarray], bucket_name: str, cluster_url: str = 'couchbase://localhost', username: str = 'Administrator', password: str = 'indoxmain'):
        from couchbase.cluster import Cluster
        from couchbase.auth import PasswordAuthenticator
        from couchbase.exceptions import CouchbaseException
        from couchbase.bucket import Bucket
        try:
            self.cluster = Cluster(cluster_url, authenticator=PasswordAuthenticator(username, password))
            self.bucket = self.cluster.bucket(bucket_name)
            self.collection = self.bucket.default_collection()
            self.embedding_function = embedding_function
        except CouchbaseException as e:
            raise RuntimeError(f"Failed to initialize Couchbase connection: {e}")

    def add(self, docs: List[str]):
        from couchbase.exceptions import CouchbaseException

        if not isinstance(docs, list):
            raise ValueError("The 'docs' argument must be a list of strings")

        try:
            for doc in docs:
                embedding = self.embedding_function.embed_documents(doc)

                doc_id = str(uuid.uuid4())
                doc_data = {
                    'id': doc_id,
                    'embedding': embedding,
                    'text': doc
                }
                self.collection.upsert(doc_id, doc_data)
        except CouchbaseException as e:
            raise RuntimeError(f"Failed to add documents to Couchbase: {e}")

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        from couchbase.exceptions import CouchbaseException
        from couchbase.search import SearchOptions, MatchQuery
        try:
            query_embedding = self.embedding_function.embed_query(query)
    
            match_query = MatchQuery(query)
            search_options = SearchOptions(limit=k)
            result = self.cluster.search_query('FTS_QA', match_query, search_options)
    
            similarity_scores = []
            for row in result.rows():
                doc_id = row.id
    
                try:
                    doc_data = self.collection.get(doc_id).content_as[dict]
                    print(f"Document with ID {doc_id}: {json.dumps(doc_data, indent=2)}")
    
                    if 'answer' in doc_data:
                        document_content = doc_data['answer']
                    elif 'question' in doc_data:
                        document_content = doc_data['question']
                    else:
                        document_content = doc_data.get('title', '') + ' ' + doc_data.get('name', '') + ' ' + doc_data.get('address', '')
    
                    document = Document(page_content=document_content, id=doc_id)
                    score = row.score
                    similarity_scores.append((document, score))
    
                except CouchbaseException as e:
                    print(f"Failed to fetch document with ID {doc_id}: {e}")
    
            similarity_scores.sort(reverse=True, key=lambda x: x[1])
            return similarity_scores[:k]
    
        except CouchbaseException as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")

