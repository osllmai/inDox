from typing import List, Dict, Any, Optional, Tuple
import uuid
from indox.core import Document

class Qdrant:
    def __init__(
            self,
            collection_name: str,
            embedding_function: Any,
            url: str,
            api_key: str,
    ):
        """
        Initialize Qdrant vectorstore.

        Args:
            collection_name (str): Name of the Qdrant collection.
            embedding_function (Any): Function to generate embeddings.
            url (str): URL of Qdrant service.
            api_key (str): API key for Qdrant cloud.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
        except ImportError:
            raise ImportError("Could not import qdrant-client python package. "
                              "Please install it with `pip install qdrant-client`.")

        self._embedding_function = embedding_function
        self._collection_name = collection_name
        self._client = QdrantClient(url=url, api_key=api_key)
        self._create_collection()

    def _create_collection(self):
        """Create the collection with the embedding size."""
        from qdrant_client.http import models as rest

        embedding_size = self._get_embedding_size()

        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=rest.VectorParams(
                    size=embedding_size,
                    distance=rest.Distance.COSINE
                )
            )
            print(f"Collection {self._collection_name} created with vector size {embedding_size}.")
        except Exception as e:
            if "already exists" in str(e):
                print(f"Collection {self._collection_name} already exists.")
            else:
                raise e

    def _get_embedding_size(self) -> int:
        """Get the size of embeddings produced by the embedding function."""
        sample_embedding = self._embedding_function.embed_query("sample text")
        return len(sample_embedding)

    def add(
            self,
            texts: List[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to the vectorstore.

        Args:
            texts (List[str]): The texts to add.
            metadatas (Optional[List[dict]]): Metadata for each text.
            ids (Optional[List[str]]): IDs for each text.

        Returns:
            List[str]: The IDs of the added texts.
        """
        from qdrant_client.http import models as rest

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding_function.embed_documents(texts)

        payloads = [
            {
                "page_content": text,
                "metadata": metadata or {}
            }
            for text, metadata in zip(texts, metadatas or [{}] * len(texts))
        ]

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                rest.PointStruct(
                    id=id_,
                    vector=embedding,
                    payload=payload
                )
                for id_, embedding, payload in zip(ids, embeddings, payloads)
            ]
        )

        return ids

    def _similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with scores.

        Args:
            query (str): The query text.
            k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Tuple[Document, float]]: The search results with scores.
        """
        from qdrant_client.http import models as rest

        query_vector = self._embedding_function.embed_query(query)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=rest.Filter(**filter) if filter else None
        )

        return [
            (
                Document(
                    page_content=result.payload["page_content"],
                    metadata=result.payload["metadata"]
                ),
                result.score
            )
            for result in results
        ]