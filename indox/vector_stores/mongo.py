from typing import Optional, Dict, Any, List, Tuple, Iterable, Union
from indox.core import VectorStore, Embeddings, Document
import uuid


# import pymongo
# from pymongo.collection import Collection


class MongoDB:
    DEFAULT_K = 4
    """
    A vector store implementation using MongoDB for document storage and retrieval.

    This class provides methods to add, retrieve, update, and delete documents in a MongoDB collection.
    It supports vector similarity search using cosine similarity when an embedding function is provided.

    Attributes:
        _client: MongoDB client instance.
        _db: MongoDB database instance.
        _collection: MongoDB collection instance.
        _embedding_function: Function to generate embeddings for documents.
        _index_name: Name of the index used for similarity search.
        _text_key: Key used to store document text in the collection.
        _embedding_key: Key used to store document embeddings in the collection.
    """

    def __init__(
            self,
            collection_name: str,
            embedding_function: Optional[Embeddings] = None,
            connection_string: str = "mongodb://localhost:27017/",
            database_name: str = "vector_db",
            index_name: str = "default",
            text_key: str = "text",
            embedding_key: str = "embedding",
    ) -> None:
        """
        Initialize the MongoDB vector store.

        Args:
            collection_name: Name of the MongoDB collection to use.
            embedding_function: Optional function to generate embeddings for documents.
            connection_string: MongoDB connection string.
            database_name: Name of the MongoDB database to use.
            index_name: Name of the index used for similarity search.
            text_key: Key used to store document text in the collection.
            embedding_key: Key used to store document embeddings in the collection.
        """
        from pymongo.errors import ConnectionFailure, ConfigurationError
        try:
            from pymongo import MongoClient
            self._client = MongoClient(connection_string)
            self._db = self._client[database_name]
            self._collection = self._db[collection_name]
        except ConnectionFailure:
            raise ConnectionFailure(f"Failed to connect to MongoDB server at {connection_string}")
        except ConfigurationError as e:
            raise ConfigurationError(f"MongoDB configuration error: {str(e)}")

        if not embedding_function:
            raise ValueError("Embedding function must be provided")
        self._embedding_function = embedding_function
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store.

        Args:
            texts: Iterable of strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of ids for the documents.
            **kwargs: Additional keyword arguments.

        Returns:
            List of ids for the added documents.
        """
        from pymongo.errors import BulkWriteError

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")

        embeddings = None
        if self._embedding_function:
            try:
                embeddings = self._embedding_function.embed_documents(list(texts))
            except Exception as e:
                raise ValueError(f"Error generating embeddings: {str(e)}")

        to_insert = []
        for i, text in enumerate(texts):
            doc = {
                "_id": ids[i],
                self._text_key: text,
            }
            if embeddings:
                doc[self._embedding_key] = embeddings[i]
            if metadatas and i < len(metadatas):
                doc["metadata"] = metadatas[i]
            to_insert.append(doc)

        try:
            self._collection.insert_many(to_insert)
        except BulkWriteError as e:
            raise BulkWriteError(f"Error inserting documents: {str(e)}")

        return ids

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            **kwargs: Additional keyword arguments.

        Returns:
            List of ids for the added documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    def similarity_search_with_score(
            self,
            query: str,
            k: int = DEFAULT_K,
            filter: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search and return documents with scores.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional filter to apply to the search.
            **kwargs: Additional keyword arguments.

        Returns:
            List of tuples containing Document objects and their similarity scores.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        if not self._embedding_function:
            raise ValueError("Embedding function is not set")

        try:
            query_embedding = self._embedding_function.embed_query(query)
        except Exception as e:
            raise ValueError(f"Error generating query embedding: {str(e)}")

        try:
            if filter:
                docs = list(self._collection.find(filter, {self._embedding_key: 1, self._text_key: 1, "metadata": 1}))
            else:
                docs = list(self._collection.find({}, {self._embedding_key: 1, self._text_key: 1, "metadata": 1}))
        except Exception as e:
            raise RuntimeError(f"Error querying MongoDB: {str(e)}")

        if not docs:
            return []

        doc_embeddings = [doc[self._embedding_key] for doc in docs]
        similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()

        scored_docs = list(zip(docs, similarities))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_docs[:k]

        return [(Document(page_content=doc[self._text_key], metadata=doc.get("metadata", {})), score) for doc, score in
                top_k]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: Optional list of document ids to delete.
            **kwargs: Additional keyword arguments.
        """
        if ids:
            self._collection.delete_many({"_id": {"$in": ids}})
        else:
            self._collection.delete_many({})

    def delete_collection(self) -> None:
        """Delete the entire collection from the database."""
        self._db.drop_collection(self._collection.name)

    def __len__(self) -> int:
        return self._collection.count_documents({})

    def get(
            self,
            ids: Optional[Union[str, List[str]]] = None,
            filter: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None,
            skip: Optional[int] = None,
            text_search: Optional[str] = None,
            include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents from the vector store.

        Args:
            ids: Optional id or list of ids to retrieve.
            filter: Optional filter to apply.
            limit: Optional limit on the number of results.
            skip: Optional number of documents to skip.
            text_search: Optional text to search for.
            include: Optional list of fields to include in the results.

        Returns:
            Dictionary containing retrieved document information.
        """
        query = {}
        if ids:
            if isinstance(ids, str):
                query["_id"] = ids
            else:
                query["_id"] = {"$in": ids}

        if filter:
            query.update(filter)

        if text_search:
            query["$text"] = {"$search": text_search}

        projection = {"_id": 1}
        if include:
            if "embeddings" in include:
                projection[self._embedding_key] = 1
            if "metadata" in include:
                projection["metadata"] = 1
            if "text" in include:
                projection[self._text_key] = 1
        else:
            projection["metadata"] = 1
            projection[self._text_key] = 1

        cursor = self._collection.find(query, projection)

        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        results = list(cursor)
        return {
            "ids": [str(doc["_id"]) for doc in results],
            "embeddings": [doc.get(self._embedding_key) for doc in results] if "embeddings" in (
                    include or []) else None,
            "metadata": [doc.get("metadata") for doc in results],
            "documents": [doc.get(self._text_key) for doc in results],
        }

    def update_document(self, document_id: str, document: Document) -> None:
        """
        Update a single document in the vector store.

        Args:
            document_id: Id of the document to update.
            document: New Document object to replace the existing one.
        """
        self.update_documents([document_id], [document])

    def update_documents(self, ids: List[str], documents: List[Document]) -> None:
        """
        Update multiple documents in the vector store.

        Args:
            ids: List of document ids to update.
            documents: List of new Document objects to replace the existing ones.
        """
        if self._embedding_function is None:
            raise ValueError("For update, you must specify an embedding function on creation.")

        if len(ids) != len(documents):
            raise ValueError("Number of ids must match number of documents")

        text = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]
        try:
            embeddings = self._embedding_function.embed_documents(text)
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")

        for i, doc_id in enumerate(ids):
            update_data = {
                self._text_key: text[i],
                self._embedding_key: embeddings[i],
                "metadata": metadata[i]
            }
            try:
                result = self._collection.update_one({"_id": doc_id}, {"$set": update_data})
                if result.modified_count == 0:
                    raise ValueError(f"No document found with id {doc_id}")
            except Exception as e:
                raise RuntimeError(f"Error updating document {doc_id}: {str(e)}")
