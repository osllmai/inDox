import json
import uuid
from typing import List, Optional, Any, Iterable, Type
import logging
from indox.core import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class DuckDB:

    def __init__(self,
                 embedding_function: Any = None,
                 vector_key: str = "embedding",
                 id_key: str = "id",
                 text_key: str = "text",
                 table_name: str = "embeddings",
                 ):

        try:
            import duckdb
        except ImportError:
            raise ImportError("Could not import duckdb package. Please install it with `pip install duckdb`.")
        self.duckdb = duckdb
        self._connection = self.duckdb.connect(database=":memory:",
                                                             config={"enable_external_access": "false"})
        self._embedding_function = embedding_function
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key
        self._table_name = table_name

        if self._embedding_function is None:
            raise ValueError("An embedding function must be provided.")

        self._ensure_table()
        self._table = self._connection.table(self._table_name)

    def _ensure_table(self):
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            {self._id_key} VARCHAR PRIMARY KEY,
            {self._text_key} VARCHAR,
            {self._vector_key} FLOAT[],
            metadata VARCHAR
        )
        """
        self._connection.execute(create_table_sql)

    def add(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Turn texts into embeddings and add them to the database using Pandas DataFrame.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: Additional parameters including optional 'ids' to associate with the texts.

        Returns:
            List of ids of the added texts.
        """
        have_pandas = False
        try:
            import pandas as pd
            have_pandas = True
        except ImportError:
            logger.info(
                "Unable to import pandas. "
                "Install it with `pip install -U pandas` "
                "to improve performance of add_texts()."
            )

        ids = kwargs.pop("ids", [str(uuid.uuid4()) for _ in texts])

        embeddings = self._embedding_function.embed_documents(list(texts))

        data = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = json.dumps(metadatas[idx]) if metadatas and idx < len(metadatas) else None

            if have_pandas:
                data.append({
                    self._id_key: ids[idx],
                    self._text_key: text,
                    self._vector_key: embedding,
                    "metadata": metadata,
                })
            else:
                self._connection.execute(
                    f"INSERT INTO {self._table_name} VALUES (?, ?, ?, ?)",
                    [ids[idx], text, embedding, metadata]
                )

        if have_pandas:
            df = pd.DataFrame(data)
            self._connection.register("df", df)
            self._connection.execute(
                f"INSERT INTO {self._table_name} SELECT * FROM df"
            )

        return ids

    def get_all_data(self):
        query = f"SELECT * FROM {self._table_name}"
        result = self._connection.execute(query).fetchall()
        return result

    # @classmethod
    # def from_texts(
    #         cls: Type["DuckDB"],
    #         texts: List[str],
    #         embedding: Any,
    #         metadatas: Optional[List[dict]] = None,
    #         **kwargs: Any,
    # ) -> "DuckDB":
    #     """Creates an instance of DuckDB and populates it with texts and their embeddings.
    #
    #     Args:
    #         texts: List of strings to add to the vector store.
    #         embedding: The embedding function or model to use for generating embeddings.
    #         metadatas: Optional list of metadata dictionaries associated with the texts.
    #         kwargs: Additional keyword arguments including:
    #             - connection: DuckDB connection. If not provided, a new connection will be created.
    #             - vector_key: The column name for storing vectors. Default "vector".
    #             - id_key: The column name for storing unique identifiers. Default "id".
    #             - text_key: The column name for storing text. Defaults to "text".
    #             - table_name: The name of the table to use for storing embeddings. Defaults to "embeddings".
    #
    #     Returns:
    #         An instance of DuckDB with the provided texts and their embeddings added.
    #     """
    #
    #     connection = kwargs.get("connection", None)
    #     vector_key = kwargs.get("vector_key", "embedding")
    #     id_key = kwargs.get("id_key", "id")
    #     text_key = kwargs.get("text_key", "text")
    #     table_name = kwargs.get("table_name", "embeddings")
    #
    #     instance = cls(
    #         embedding_function=embedding,
    #         vector_key=vector_key,
    #         id_key=id_key,
    #         text_key=text_key,
    #         table_name=table_name,
    #     )
    #
    #     if connection is not None:
    #         instance._connection = connection
    #
    #     instance.add(
    #         texts=texts,
    #         metadatas=metadatas,
    #         **kwargs
    #     )
    #
    #     return instance

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Performs a similarity search for a given query string.
        Args:
            query: The query string to search for.
            k: The number of similar texts to return.
        Returns:
            A list of Documents most similar to the query.
        """
        embedding = self._embedding_function.embed_query(query)

        list_cosine_similarity = self.duckdb.FunctionExpression(
            "list_cosine_similarity",
            self.duckdb.ColumnExpression(self._vector_key),
            self.duckdb.ConstantExpression(embedding),
        )

        docs = (
            self._table.select(
                *[
                    self.duckdb.StarExpression(exclude=[]),
                    list_cosine_similarity.alias("similarity_score"),
                ]
            )
            .order("similarity_score desc")
            .limit(k)
            .fetchdf()
        )

        return [
            Document(
                page_content=docs[self._text_key][idx],
                metadata={
                    **json.loads(docs["metadata"][idx]),
                    "Similarity Score": docs['similarity_score'][idx],
                }
                if docs["metadata"][idx]
                else {"Similarity Score": docs['similarity_score'][idx]},
            )
            for idx in range(len(docs))
        ]

    def delete(self, ids: List[str]) -> None:
        """Deletes records from the database based on the given list of ids.

        Args:
            ids: List of unique identifiers of the records to be deleted.
        """
        if not ids:
            logger.warning("No IDs provided for deletion.")
            return

        placeholders = ','.join(['?'] * len(ids))
        delete_sql = f"DELETE FROM {self._table_name} WHERE {self._id_key} IN ({placeholders})"

        try:
            self._connection.execute(delete_sql, ids)
            logger.info(f"Successfully deleted {len(ids)} records.")
        except Exception as e:
            logger.error(f"Failed to delete records: {e}")
            raise