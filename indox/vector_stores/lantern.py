# import psycopg2
# import psycopg2.extras
# import uuid
# import json
# from typing import Optional, List, Dict, Any, Tuple
# from indox.core import Document
#
# class LanternDB:
#     DEFAULT_K = 4
#
#
#     def __init__(
#             self,
#             collection_name: str,
#             embedding_function: Optional[Any] = None,
#             connection_params: Dict[str, str] = None,
#             dimension: int = 768
#     ) -> None:
#         """
#         Initialize the LanternDBClient class, connecting to LanternDB.
#
#         Args:
#             collection_name: Name of the collection to use in LanternDB.
#             embedding_function: Function to generate embeddings.
#             connection_params: Dictionary containing connection parameters for LanternDB.
#             dimension: The dimensionality of the vector embeddings.
#         """
#         self._collection_name = collection_name
#         self._embedding_function = embedding_function
#         self._dimension = dimension
#         self._text_key = "text"
#         self._vector_key = "embedding"
#
#         self._conn = psycopg2.connect(**connection_params)
#
#         print(f"Connected to LanternDB collection '{self._collection_name}'")
#         self._cursor = self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
#         self._ensure_collection()
#
#     def _ensure_collection(self):
#         """
#         Ensure the collection exists in LanternDB.
#         """
#         try:
#             self._cursor.execute(f"""
#                 CREATE TABLE IF NOT EXISTS {self._collection_name} (
#                     id TEXT PRIMARY KEY,
#                     {self._text_key} TEXT,
#                     {self._vector_key} VECTOR({self._dimension}),
#                     metadata JSONB
#                 )
#             """)
#             self._conn.commit()
#             print(f"Collection '{self._collection_name}' created successfully.")
#         except Exception as e:
#             self._conn.rollback()
#             raise RuntimeError(f"Error ensuring collection: {e}")
#
#     def add_texts(
#             self,
#             texts: List[str],
#             metadatas: Optional[List[dict]] = None,
#             ids: Optional[List[str]] = None
#     ) -> List[str]:
#         """
#         Add texts and their embeddings to LanternDB.
#         """
#         if not ids:
#             ids = [str(uuid.uuid4()) for _ in texts]
#
#         if not self._embedding_function:
#             raise ValueError("Embedding function must be provided")
#
#         embeddings = self._embedding_function.embed_documents(texts)
#
#         try:
#             for i, text in enumerate(texts):
#                 metadata = metadatas[i] if metadatas else {}
#                 self._cursor.execute(f"""
#                     INSERT INTO {self._collection_name} (id, {self._text_key}, {self._vector_key}, metadata)
#                     VALUES (%s, %s, %s, %s)
#                 """, (ids[i], text, embeddings[i], json.dumps(metadata)))
#
#             self._conn.commit()
#             print(f"Inserted {len(texts)} documents into collection '{self._collection_name}'.")
#         except Exception as e:
#             self._conn.rollback()
#             print(f"Error inserting documents: {e}")
#
#         return ids
#
#     def _similarity_search_with_score(
#             self,
#             query: str,
#             k: int = DEFAULT_K,
#             **kwargs: Any
#     ) -> List[Tuple[Document, float]]:
#         """
#         Perform a similarity search in LanternDB and return documents with their scores.
#
#         Args:
#             query: The query string to search for.
#             k: The number of similar texts to return.
#
#         Returns:
#             A list of tuples where each tuple contains a Document object and its similarity score.
#         """
#         if not self._embedding_function:
#             raise ValueError("Embedding function is not set")
#
#         query_embedding = self._embedding_function.embed_query(query)
#
#         try:
#             self._cursor.execute(f"""
#                 SELECT {self._text_key}, metadata, 1 - ({self._vector_key} <-> %s) AS similarity_score
#                 FROM {self._collection_name}
#                 ORDER BY similarity_score DESC
#                 LIMIT %s
#             """, (query_embedding, k))
#
#             results = self._cursor.fetchall()
#
#             return [
#                 (
#                     Document(
#                         page_content=row[self._text_key],
#                         metadata={
#                             **json.loads(row['metadata']),
#                             "Similarity Score": row['similarity_score'],
#                         } if row['metadata'] else {"Similarity Score": row['similarity_score']},
#                     ),
#                     row['similarity_score']
#                 )
#                 for row in results
#             ]
#         except Exception as e:
#             print(f"Error performing similarity search: {e}")
#             return []
#
#     def delete(self, ids: Optional[List[str]] = None):
#         """Delete documents from the LanternDB collection."""
#         try:
#             if ids:
#                 self._cursor.execute(f"""
#                     DELETE FROM {self._collection_name}
#                     WHERE id = ANY(%s)
#                 """, (ids,))
#                 self._conn.commit()
#                 print(f"Deleted {len(ids)} documents from collection '{self._collection_name}'.")
#             else:
#                 self._cursor.execute(f"DROP TABLE IF EXISTS {self._collection_name}")
#                 self._conn.commit()
#                 print(f"Deleted entire collection '{self._collection_name}'.")
#         except Exception as e:
#             self._conn.rollback()
#             print(f"Error deleting documents: {e}")
#
#     def __len__(self) -> int:
#         """Return the count of documents in the LanternDB collection."""
#         try:
#             self._cursor.execute(f"SELECT COUNT(*) FROM {self._collection_name}")
#             count = self._cursor.fetchone()[0]
#             return count
#         except Exception as e:
#             print(f"Error getting document count: {e}")
#             return 0
#
#     def __del__(self):
#         """Close the database connection when the object is destroyed."""
#         if hasattr(self, '_cursor'):
#             self._cursor.close()
#         if hasattr(self, '_conn'):
#             self._conn.close()
