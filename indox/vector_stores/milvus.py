import os
from tqdm import tqdm
import json
import logging
from pymilvus import MilvusClient, MilvusException
from typing import Optional, List, Tuple
from indox.core import VectorStore, Embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure that ANSI color codes are not included in the log output
for handler in logging.root.handlers:
    handler.addFilter(lambda record: hasattr(handler, 'stream') and handler.stream.isatty())

class Milvus:
    def __init__(self, embedding_model: Embeddings, collection_name: str = "indox_collection"):
        """Initialize the Milvus instance.

        Args:
            embedding_model: The model used for generating embeddings.
            collection_name: The name of the Milvus collection (default is 'indox_collection').
        """
        try:
            self.milvus_client = MilvusClient(host='127.0.0.1', port='19530')
            self.collection_name = collection_name
            self.embedding_dim = None
            self.embedding_model = embedding_model
            logger.info(f"Initialized Milvus with collection '{collection_name}'")
        except MilvusException as e:
            logger.error(f"Failed to initialize Milvus client: {str(e)}")
            raise

    def load_text_lines(self, file_path: str) -> List[str]:
        """Load text lines from a file.

        Args:
            file_path: The path to the text file.

        Returns:
            A list of text lines.
        """
        try:
            with open(file_path, 'r') as file:
                text_lines = file.readlines()
            text_lines = [line.strip() for line in text_lines]
            logger.info(f"Loaded {len(text_lines)} lines from {file_path}")
            return text_lines
        except Exception as e:
            logger.error(f"Error loading text lines from {file_path}: {str(e)}")
            raise

    def emb_text(self, text: str) -> List[float]:
        """Generate an embedding for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A list representing the embedding vector.
        """
        try:
            embedding = self.embedding_model.embed_query(text)
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
            logger.info(f"Generated embedding for text: '{text}'")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: '{text}': {str(e)}")
            raise

    def create_collection(self) -> None:
        """Create a Milvus collection."""
        try:
            if self.embedding_dim is None:
                raise ValueError("Embedding dimension is not set. Ensure that you generate an embedding first.")

            if self.milvus_client.has_collection(self.collection_name):
                self.milvus_client.drop_collection(self.collection_name)

            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim,
                metric_type="IP",  # Inner Product
                consistency_level="Strong",
            )
            logger.info(f"Created collection '{self.collection_name}' with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error creating collection '{self.collection_name}': {str(e)}")
            raise

    def insert_data(self, text_lines: List[str]) -> None:
        """Insert a list of text lines into the Milvus collection.

        Args:
            text_lines: The list of text lines to insert.

        Returns:
            None
        """
        try:
            data = [{"id": i, "vector": self.emb_text(line), "text": line} for i, line in
                    enumerate(tqdm(text_lines, desc="Creating embeddings"))]
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"Inserted {len(text_lines)} items into collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error inserting data into collection '{self.collection_name}': {str(e)}")
            raise

    def similarity_search_with_score(self, question: str, limit: int = 3, k: Optional[int] = None) -> List[
        Tuple[str, float]]:
        """Perform a similarity search in the Milvus collection for a given question.

        Args:
            question: The question to search for.
            limit: The maximum number of results to return (default is 3).

        Returns:
            A list of tuples, each containing a retrieved text and its similarity score.
        """
        try:
            search_res = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[self.emb_text(question)],
                limit=limit,
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["text"],
            )
            logger.info(f"Performed similarity search for question: '{question}'")
            return [
                (res["entity"]["text"], res["distance"])
                for res in search_res[0]
            ]
        except Exception as e:
            logger.error(f"Error performing similarity search for question '{question}': {str(e)}")
            raise

    def add_documents(self, documents: List[str]) -> List[str]:
        """Add more documents to the vector store by generating embeddings and inserting them.

        Args:
            documents: The documents to add to the vector store.

        Returns:
            A list of IDs of the added texts.
        """
        try:
            new_ids = []
            for i, doc in enumerate(documents):
                embedding = self.emb_text(doc)
                doc_id = str(len(new_ids))  # Generate a new ID
                self.milvus_client.insert(
                    collection_name=self.collection_name,
                    data=[{"id": doc_id, "vector": embedding, "text": doc}]
                )
                new_ids.append(doc_id)
            logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")
            return new_ids
        except Exception as e:
            logger.error(f"Error adding documents to collection '{self.collection_name}': {str(e)}")
            raise

    def add_texts(self, texts: List[str]) -> None:
        """Add documents to the Milvus collection by generating embeddings and inserting them.

        Args:
            texts: The list of documents to add to the vector store.

        Returns:
            None
        """
        try:
            data = [{"id": str(i), "vector": self.emb_text(doc), "text": doc} for i, doc in enumerate(texts)]
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"Added {len(texts)} documents to collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection in Milvus.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.milvus_client.has_collection(self.collection_name):
                self.milvus_client.drop_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection '{self.collection_name}': {str(e)}")
            raise

    def update_document(self, document_id: str, document: str) -> None:
        """Update an existing document in the Milvus collection.

        Args:
            document_id: The ID of the document to update.
            document: The new content for the document.

        Returns:
            None
        """
        try:
            embedding = self.emb_text(document)
            self.milvus_client.update(
                collection_name=self.collection_name,
                data=[{"id": document_id, "vector": embedding, "text": document}]
            )
            logger.info(f"Updated document with ID '{document_id}' in collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error updating document ID '{document_id}' in collection '{self.collection_name}': {str(e)}")
            raise

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """Delete documents from the Milvus collection by their IDs.

        Args:
            ids: The IDs of the documents to delete (default is None).

        Returns:
            None
        """
        try:
            if ids is not None:
                self.milvus_client.delete(
                    collection_name=self.collection_name,
                    ids=ids
                )
            logger.info(f"Deleted documents with IDs: {ids} from collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting documents from collection '{self.collection_name}': {str(e)}")
            raise

    def __len__(self) -> int:
        """Count the number of documents in the Milvus collection.

        Args:
            None

        Returns:
            int: The number of documents in the collection.
        """
        try:
            count = self.milvus_client.count(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' contains {count} documents")
            return count
        except Exception as e:
            logger.error(f"Error counting documents in collection '{self.collection_name}': {str(e)}")
            raise
