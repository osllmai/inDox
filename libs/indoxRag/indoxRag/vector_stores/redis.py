from indoxRag.core import Document
from loguru import logger
import sys
import numpy as np
from typing import List, Tuple, Union
import json
import uuid

logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class RedisDB:
    """
    A concrete implementation of VectorStore using Redis for storage.

    Attributes:
        redis_client (redis.Redis): The Redis client.
        embedding (callable): The embedding function to use.
        prefix (str): The prefix for Redis keys.
    """

    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        embedding: callable,
        prefix: str = "doc",
    ):
        """
        Initializes the RedisVector.

        Args:
            host (str): The host of the Redis server.
            port (int): The port of the Redis server.
            password (str): The password for the Redis server.
            embedding (callable): The embedding function to use.
            prefix (str, optional): The prefix for Redis keys. Defaults to "doc".
        """
        import redis

        self.redis_client = redis.Redis(host=host, port=port, password=password)
        self.embedding = embedding
        self.prefix = prefix

    def add(
        self, texts: List[str], metadatas: Union[List[dict], None] = None
    ) -> List[str]:
        """
        Adds texts to the Redis vector store.

        Args:
            texts (List[str]): The texts to be added to the vector store.
            metadatas (List[dict], optional): Metadata for each text. Defaults to None.

        Returns:
            List[str]: List of IDs for the added texts.
        """
        try:
            ids = [str(uuid.uuid4()) for _ in texts]
            embeddings = self.embedding.embed_documents(texts)

            pipe = self.redis_client.pipeline()
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                key = f"{self.prefix}:{ids[i]}"
                metadata = json.dumps(metadatas[i] if metadatas else {})

                # Store the text and its embedding
                pipe.hset(
                    key,
                    mapping={
                        "content": text,
                        "embedding": json.dumps(embedding),
                        "metadata": metadata,
                    },
                )

                pipe.sadd(f"{self.prefix}:keys", key)

            pipe.execute()
            logger.info(f"Added {len(texts)} texts successfully to the vector store.")
            return ids
        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise RuntimeError(f"Can't add texts to the vector store: {e}")

    def _similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves documents similar to the given query from the Redis vector store.

        Args:
            query (str): The query to retrieve similar documents.
            k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            List[Tuple[Document, float]]: A list of tuples containing the similar documents and their scores.
        """
        try:
            query_embedding = self.embedding.embed_query(query)

            all_keys = self.redis_client.smembers(f"{self.prefix}:keys")
            results = []

            pipe = self.redis_client.pipeline()
            for key in all_keys:
                pipe.hgetall(key)
            all_docs = pipe.execute()

            for doc_data in all_docs:
                doc_embedding = np.array(json.loads(doc_data[b"embedding"]))

                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )

                metadata = (
                    json.loads(doc_data[b"metadata"].decode())
                    if b"metadata" in doc_data
                    else {}
                )
                results.append(
                    (
                        Document(
                            page_content=doc_data[b"content"].decode(),
                            metadata=metadata,
                        ),
                        similarity,
                    )
                )

            # Sort results by similarity score (descending) and return top k
            return sorted(results, key=lambda x: x[1], reverse=True)[:k]

        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise RuntimeError(f"Can't perform similarity search: {e}")

    def delete(self, ids: List[str]) -> None:
        """
        Deletes documents from the Redis vector store.

        Args:
            ids (List[str]): The list of document IDs to delete.
        """
        try:
            pipe = self.redis_client.pipeline()
            for id in ids:
                pipe.delete(f"{self.prefix}:{id}")
                pipe.srem(f"{self.prefix}:keys", f"{self.prefix}:{id}")
            pipe.execute()
            logger.info(f"Successfully deleted {len(ids)} documents.")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RuntimeError(f"Can't delete documents from the vector store: {e}")
