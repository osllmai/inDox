from typing import Optional, Dict, Any, List, Tuple, Iterable, Union
from indoxArcg.core import Embeddings, Document
import uuid
import os


class PineconeVectorStore:
    DEFAULT_K = 4
    """
    A vector store implementation using Pinecone as the backend.

    This class provides methods to add documents, perform similarity searches,
    and manage the Pinecone index. It uses an embedding function to convert
    text into vector representations.

    Attributes:
        DEFAULT_K (int): The default number of results to return in similarity searches.

    Args:
        index_name (str): The name of the Pinecone index to use.
        embedding_function (Optional[Embeddings]): The function to use for embedding documents and queries.
        text_key (str): The key used to store the document text in Pinecone metadata.
        namespace (Optional[str]): The namespace to use in the Pinecone index.
        pinecone_api_key (Optional[str]): The Pinecone API key. If not provided, it will be read from the PINECONE_API_KEY environment variable.

    Raises:
        ImportError: If the Pinecone Python package is not installed.
        ValueError: If the embedding function is not provided or if the Pinecone API key is missing.
    """

    def _flatten_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool, List[str]]]:
        """
        Flatten metadata to ensure it only contains types supported by Pinecone.

        Args:
            metadata (Dict[str, Any]): Original metadata.

        Returns:
            Dict[str, Union[str, int, float, bool, List[str]]]: Flattened metadata.
        """
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or (
                isinstance(value, list) and all(isinstance(i, str) for i in value)
            ):
                flattened[key] = value
            else:
                flattened[key] = str(value)
        return flattened

    def __init__(
        self,
        index_name: str,
        embedding_function: Optional[Embeddings] = None,
        text_key: str = "content",
        namespace: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
    ) -> None:
        try:
            from pinecone import Pinecone as PineconeClient
        except ImportError:
            raise ImportError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone`."
            )

        if not embedding_function:
            raise ValueError("Embedding function must be provided")

        self._embedding_function = embedding_function
        self._text_key = text_key
        self._namespace = namespace

        pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "Pinecone API key must be provided as an argument or set as PINECONE_API_KEY environment variable."
            )

        self._client = PineconeClient(api_key=pinecone_api_key)
        self._index = self._client.Index(index_name)

    @classmethod
    def create_index(
        cls,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        pinecone_api_key: Optional[str] = None,
    ) -> None:
        """
        Creates a new serverless Pinecone index.

        This class method initializes a new serverless Pinecone index with the specified
        parameters. It uses the Pinecone API to create the index in the specified cloud
        and region.

        Args:
            index_name (str): The name of the index to create.
            dimension (int): The dimension of the vectors to be stored in the index.
            metric (str, optional): The distance metric to use for similarity search. Defaults to "cosine".
            cloud (str, optional): The cloud provider to use. Defaults to "aws".
            region (str, optional): The region in which to create the index. Defaults to "us-east-1".
            pinecone_api_key (Optional[str], optional): The Pinecone API key. If not provided,
                it will be read from the PINECONE_API_KEY environment variable.

        Raises:
            ValueError: If the Pinecone API key is not provided and not set in the environment.
            Exception: If there's an error creating the index.

        """
        from pinecone import Pinecone, ServerlessSpec

        pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "Pinecone API key must be provided as an argument or set as PINECONE_API_KEY environment variable."
            )

        pc = Pinecone(api_key=pinecone_api_key)
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            print(f"Serverless index '{index_name}' created successfully.")
        except Exception as e:
            raise Exception(f"Error creating serverless index: {str(e)}")

    def _add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store.

        Args:
            texts (Iterable[str]): Texts to add.
            metadatas (Optional[List[dict]]): Metadata for each text.
            ids (Optional[List[str]]): IDs for each text.
            **kwargs: Additional arguments.

        Returns:
            List[str]: List of IDs of added texts.

        Raises:
            ValueError: If number of metadatas doesn't match number of texts.
            RuntimeError: If there's an error upserting to Pinecone.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")

        try:
            embeddings = self._embedding_function.embed_documents(list(texts))
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")

        to_upsert = []
        for i, (id, text, embedding) in enumerate(zip(ids, texts, embeddings)):
            metadata = self._flatten_metadata(metadatas[i] if metadatas else {})
            metadata[self._text_key] = text
            metadata["id"] = id
            to_upsert.append((id, embedding, metadata))

        try:
            self._index.upsert(vectors=to_upsert, namespace=self._namespace)
        except Exception as e:
            raise RuntimeError(f"Error upserting to Pinecone: {str(e)}")

        return ids

    def add(self, docs: List[Union[str, Document]]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            docs (List[Union[str, Document]]): Documents to add.

        Returns:
            List[str]: List of IDs of added documents.

        Raises:
            ValueError: If an invalid document type is provided.
            RuntimeError: If there's an error adding documents to the vector store.
        """
        texts = []
        metadatas = []
        for doc in docs:
            if isinstance(doc, Document):
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
            elif isinstance(doc, str):
                texts.append(doc)
                metadatas.append({})
            else:
                raise ValueError(
                    f"Invalid document type: {type(doc)}. Expected str or Document."
                )

        try:
            return self._add_texts(texts=texts, metadatas=metadatas)
        except Exception as e:
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform a similarity search.

        Args:
            query (str): Query text.
            k (int): Number of results to return.
            filter (Optional[Dict[str, Any]]): Filter to apply to the search.
            **kwargs: Additional arguments.

        Returns:
            List[Document]: List of similar documents.
        """
        docs_and_scores = self._similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def _similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search and return documents with scores.

        Args:
            query (str): Query text.
            k (int): Number of results to return.
            filter (Optional[Dict[str, Any]]): Filter to apply to the search.
            **kwargs: Additional arguments.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing similar documents and their scores.

        Raises:
            ValueError: If the embedding function is not set or there's an error generating query embedding.
            RuntimeError: If there's an error querying Pinecone.
        """
        if not self._embedding_function:
            raise ValueError("Embedding function is not set")

        try:
            query_embedding = self._embedding_function.embed_query(query)
        except Exception as e:
            raise ValueError(f"Error generating query embedding: {str(e)}")

        try:
            results = self._index.query(
                vector=query_embedding,
                top_k=k,
                namespace=self._namespace,
                filter=filter,
                include_metadata=True,
            )
        except Exception as e:
            raise RuntimeError(f"Error querying Pinecone: {str(e)}")

        docs_and_scores = []
        for match in results["matches"]:
            metadata = match["metadata"]
            text = metadata.pop(self._text_key)
            doc = Document(page_content=text, **metadata)
            docs_and_scores.append((doc, match["score"]))

        return docs_and_scores

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids (Optional[List[str]]): List of IDs to delete.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If no IDs are provided.
            RuntimeError: If there's an error deleting from Pinecone.
        """
        if ids:
            try:
                self._index.delete(ids=ids, namespace=self._namespace)
            except Exception as e:
                raise RuntimeError(f"Error deleting from Pinecone: {str(e)}")
        else:
            raise ValueError("No IDs provided for delete request")

    def __len__(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            int: Total number of vectors in the index.
        """
        stats = self._index.describe_index_stats()
        return stats["total_vector_count"]
