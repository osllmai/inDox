from typing import List, Any, Dict, Tuple
import numpy as np
from indoxRag.core import Document


class Vearch:
    def __init__(self, embedding_function: Any, db_name: str, space_name: str = "text_embedding_space"):
        
        """
        Initialize Vearch with an external embedding function, a database name, and space definition.
        
        Args:
        embedding_function: Function or model that converts text to embeddings.
        db_name: Name of the database to store documents.
        space_name: Name of the space (default: "text_embedding_space").
        """
        from vearch.config import Config
        self.embedding_function = embedding_function
        self.db_name = db_name
        self.space_name = space_name
        self.data = []
        self.dim = len(self.embedding_function.embed_query("test"))
        self.space_schema = self.create_space_schema()

        self.config = Config(host="http://localhost:9001", token="secret")


    def create_space_schema(self) :
        from vearch.schema.field import Field
        from vearch.schema.space import SpaceSchema
        from vearch.utils import DataType, MetricType
        from vearch.schema.index import FlatIndex, ScalarIndex
        """Define space schema with changed field names."""
        text_field = Field(
            "docs",
            DataType.STRING,
            desc="docs field",
            index=ScalarIndex("docs_idx"),
        )
        embedding_field = Field(
            "embedding",
            DataType.VECTOR,
            FlatIndex("embedding_idx", MetricType.Inner_product),
            dimension=self.dim,
        )
        space_schema = SpaceSchema(
            "text_embedding_space",
            fields=[text_field, embedding_field],
        )
        return space_schema

    def create_space(self, space_name: str):
        """Create a new space."""
        space_schema = self.create_space_schema()
        return self.vc.create_space(self.db_name, space_schema)

    def create_database(self, db_name: str):
        """Create a new database."""
        ret = self.vc.create_database(db_name)
        return ret

    def add(self, docs: List[Document]):
        """
        Add a list of Document objects into the Vearch database.
        
        Args:
        docs: List of Document objects containing text and metadata.
        """
        for doc in docs:
            embedding = self.embedding_function.embed_query(doc)
            self.data.append({
                "text": doc,
                "embedding": embedding,
            })
        


    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.
        
        Args:
        query: The query text to search for.
        k: Number of top results to return (default: 5).
        
        Returns:
        List of tuples with Document and similarity score.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = np.array(self.embedding_function.embed_query(query))

        results = []
        for doc in self.data:
            if 'embedding' in doc and 'text' in doc:
                doc_embedding = np.array(doc['embedding'])
                text = doc['text']

                similarity = cosine_similarity(doc_embedding.reshape(1, -1), query_embedding.reshape(1, -1))[0][0]

                document = Document(page_content=text, metadata=doc.get("metadata", {}))

                results.append((document, similarity))

        if not results:
            raise ValueError("No document embeddings found.")
        
        results = sorted(results, key=lambda x: x[1], reverse=True)

        return results[:k]
