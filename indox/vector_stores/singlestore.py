import json
from typing import List, Tuple, Optional, Any
import numpy as np
from indox.core import Document


class SingleStoreVectorDB:
    """
    A class for managing vector embeddings in a SingleStore database.

    This class provides methods for creating, updating, and querying a table
    that stores text content along with their vector representations.
    """

    def __init__(
            self,
            connection_params: dict,
            table_name: str = "embeddings",
            embedding_function: Any = None,
            vector_dimension: int = 768,
            use_vector_index: bool = True,
            use_full_text_search: bool = False,
            vector_index_options: Optional[dict] = None
    ):
        """
        Initialize the SingleStoreVectorDB instance.

        Args:
            connection_params (dict): Database connection parameters.
            table_name (str, optional): Name of the table to store embeddings. Defaults to "embeddings".
            embedding_function (Any, optional): Function to create the vector representations. Defaults to None.
            vector_dimension (int, optional): Dimension of the vector embeddings. Defaults to 768.
            use_vector_index (bool, optional): Whether to use vector indexing. Defaults to True.
            use_full_text_search (bool, optional): Whether to enable full-text search. Defaults to False.
            vector_index_options (dict, optional): Additional options for vector indexing. Defaults to None.
        """
        import singlestoredb as s2
        try:
            self.conn = s2.connect(**connection_params)
            self.table_name = table_name
            self.vector_dimension = vector_dimension
            self.use_vector_index = use_vector_index
            self.use_full_text_search = use_full_text_search
            self.vector_index_options = vector_index_options or {}

            self.embedding_function = embedding_function
            if self.embedding_function is None:
                raise ValueError("An embedding function must be provided.")

            self.id_field = "id"
            self.content_field = "content"
            self.vector_field = "vector"
            self.metadata_field = "metadata"
            self.vector_index_name = f"idx_{self.table_name}_vector"

            self._create_or_update_table()

        except s2.Error as e:
            print(f"Error initializing SingleStoreVectorDB: {e}")
            raise

    def _create_or_update_table(self):
        """
        Create the table if it doesn't exist, or update it if it does.
        """
        import singlestoredb as s2
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SHOW TABLES LIKE '{self.table_name}'")
                table_exists = cur.fetchone() is not None

                if not table_exists:
                    self._create_table(cur)
                else:
                    self._update_table_if_needed(cur)
        except s2.Error as e:
            print(f"Error creating or updating table: {e}")
            raise

    def _create_table(self, cur):
        """
        Create the table for storing embeddings.

        Args:
            cur: Database cursor.
        """
        import singlestoredb as s2

        try:
            full_text_index = ""
            if self.use_full_text_search:
                full_text_index = f", FULLTEXT({self.content_field})"

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {self.id_field} BIGINT AUTO_INCREMENT PRIMARY KEY,
                    {self.content_field} LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
                    {self.vector_field} BLOB,
                    {self.metadata_field} JSON
                    {full_text_index}
                );
            """)

            self._create_vector_index(cur)
        except s2.Error as e:
            print(f"Error creating table: {e}")
            raise

    def _update_table_if_needed(self, cur):
        """
        Update the table structure if necessary.

        Args:
            cur: Database cursor.
        """
        import singlestoredb as s2

        try:
            cur.execute(f"SHOW COLUMNS FROM {self.table_name} LIKE '{self.vector_field}'")
            vector_column_exists = cur.fetchone() is not None

            if not vector_column_exists:
                print(f"Adding missing '{self.vector_field}' column to the table.")
                cur.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {self.vector_field} BLOB")

            self._create_vector_index(cur)
        except s2.Error as e:
            print(f"Error updating table: {e}")
            raise

    def _create_vector_index(self, cur):
        """
        Create a vector index on the table if it doesn't exist.

        Args:
            cur: Database cursor.
        """
        import singlestoredb as s2

        if self.use_vector_index:
            try:
                index_options = ""
                if self.vector_index_options:
                    index_options = f"WITH ({json.dumps(self.vector_index_options)})"

                cur.execute(f"SHOW INDEX FROM {self.table_name} WHERE Key_name = '{self.vector_index_name}'")
                index_exists = cur.fetchone() is not None

                if not index_exists:
                    cur.execute(f"""
                        CREATE INDEX {self.vector_index_name} ON {self.table_name}({self.vector_field}) {index_options};
                    """)
                    print(f"Vector index '{self.vector_index_name}' created successfully.")
                else:
                    print(f"Vector index '{self.vector_index_name}' already exists.")

            except s2.Error as e:
                print(f"Warning: Unable to create vector index. This may affect search performance. Error: {e}")
                self.use_vector_index = False

    def add_texts(
            self,
            texts: List[str],
            metadatas: Optional[List[dict]] = None
    ):
        """
        Add texts to the database after embedding them.

        Args:
            texts (List[str]): List of text content to add.
            metadatas (Optional[List[dict]], optional): List of metadata for each text. Defaults to None.

        Raises:
            ValueError: If the lengths of texts and metadatas (if provided) don't match.
            s2.Error: If there's an error inserting data into the database.
        """
        import singlestoredb as s2

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError("The lengths of texts and metadatas must be the same.")

        # Use the embedding function to generate vectors from texts
        vectors = self.embedding_function.embed_documents(texts)

        try:
            with self.conn.cursor() as cur:
                for text, vector, metadata in zip(texts, vectors, metadatas):
                    cur.execute(f"""
                        INSERT INTO {self.table_name} ({self.content_field}, {self.vector_field}, {self.metadata_field})
                        VALUES (%s, %s, %s)
                    """, (text, np.array(vector, dtype=np.float32).tobytes(), json.dumps(metadata)))

            self.conn.commit()
        except s2.Error as e:
            print(f"Error adding texts to the database: {e}")
            self.conn.rollback()
            raise

    def _similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search for a given query string.

        Args:
            query (str): The query string to search for.
            k (int, optional): The number of similar texts to return. Defaults to 4.

        Returns:
            List[Tuple[Document, float]]: A list of tuples where each tuple contains a Document object and its similarity score.
        """
        # Compute the embedding for the query
        embedding = self.embedding_function.embed_query(query)
        query_vector_array = np.array(embedding, dtype=np.float32)
        query_vector_norm = np.linalg.norm(query_vector_array)
        import singlestoredb as s2

        try:
            with self.conn.cursor() as cur:
                if self.use_vector_index:
                    cur.execute(f"""
                        SELECT {self.content_field}, {self.metadata_field},
                               DOT_PRODUCT({self.vector_field}, %s) / 
                               (SQRT(DOT_PRODUCT({self.vector_field}, {self.vector_field})) * %s) as similarity
                        FROM {self.table_name}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (query_vector_array.tobytes(), float(query_vector_norm), k))
                else:
                    cur.execute(f"""
                        SELECT {self.content_field}, {self.metadata_field},
                               (SELECT SUM(a*b) / (SQRT(SUM(a*a)) * SQRT(SUM(b*b))) FROM 
                                (SELECT UNNEST(CAST({self.vector_field} AS REAL[])) as a, 
                                        UNNEST(%s::REAL[]) as b) as dots) as similarity
                        FROM {self.table_name}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (embedding, k))

                results = []
                for content, metadata, similarity in cur.fetchall():
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse metadata: {metadata}")
                            metadata = {}
                    elif metadata is None:
                        metadata = {}

                    # Construct Document object with page_content and metadata
                    document = Document(
                        page_content=content,
                        metadata={**metadata, "Similarity Score": similarity}
                    )

                    # Append (Document, similarity_score) tuple to results
                    results.append((document, float(similarity)))

            return results

        except s2.Error as e:
            print(f"Error performing similarity search: {e}")
            raise

    @staticmethod
    def delete_table(connection_params: dict, table_name: str):
        """
        Delete the specified table from the database.

        Args:
            connection_params (dict): Database connection parameters.
            table_name (str): Name of the table to delete.

        Raises:
            s2.Error: If there's an error deleting the table.
        """
        import singlestoredb as s2

        try:
            conn = s2.connect(**connection_params)
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            conn.commit()
            print(f"Table {table_name} deleted successfully.")
        except s2.Error as e:
            print(f"Error deleting table {table_name}: {e}")
            raise
        finally:
            conn.close()

