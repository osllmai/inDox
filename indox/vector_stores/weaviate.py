from __future__ import annotations

import datetime
import os
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)
from uuid import uuid4

import numpy as np
from indox.core import Document, Embeddings, VectorStore

if TYPE_CHECKING:
    import weaviate
from loguru import logger
import sys

warnings.filterwarnings("ignore")

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


def _default_schema(index_name: str, text_key: str) -> Dict:
    return {
        "class": index_name,
        "properties": [
            {
                "name": text_key,
                "dataType": ["text"],
            }
        ],
    }


def _create_weaviate_client(
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
) -> weaviate.Client:
    try:
        import weaviate
    except ImportError:
        raise ImportError(
            "Could not import weaviate python  package. "
            "Please install it with `pip install weaviate-client`"
        )
    url = url or os.environ.get("WEAVIATE_URL")
    api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
    auth = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth, **kwargs)


def _default_score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))


def _json_serializable(value: Any) -> Any:
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return value


class Weaviate:
    """`Weaviate` vector store.

    To use, you should have the ``weaviate-client`` python package installed.

    """

    def __init__(
            self,
            client: Any,
            index_name: str,
            text_key: str,
            embedding: Optional[Embeddings] = None,
            attributes: Optional[List[str]] = None,
            relevance_score_fn: Optional[
                Callable[[float], float]
            ] = _default_score_normalizer,
            by_text: bool = True,
    ):
        """Initialize with Weaviate client."""
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`."
            )
        if not isinstance(client, weaviate.Client):
            raise ValueError(
                f"client should be an instance of weaviate.Client, got {type(client)}"
            )
        self._client = client
        self._index_name = index_name
        self._embedding = embedding
        self._text_key = text_key
        self._query_attrs = [self._text_key]
        self.relevance_score_fn = relevance_score_fn
        self._by_text = by_text
        if attributes is not None:
            self._query_attrs.extend(attributes)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return (
            self.relevance_score_fn
            if self.relevance_score_fn
            else _default_score_normalizer
        )

    def _add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Upload Indox Document objects to Weaviate."""
        from weaviate.util import get_valid_uuid

        ids = []
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        embeddings: Optional[List[List[float]]] = None
        if self._embedding:
            embeddings = self._embedding.embed_documents(texts)

        with self._client.batch as batch:
            for i, text in enumerate(texts):
                data_properties = {self._text_key: text}
                if metadatas is not None:
                    for key, val in metadatas[i].items():
                        data_properties[key] = _json_serializable(val)

                # Allow for ids (consistent w/ other methods)
                _id = get_valid_uuid(uuid4())
                if "uuids" in kwargs:
                    _id = kwargs["uuids"][i]
                elif "ids" in kwargs:
                    _id = kwargs["ids"][i]

                batch.add_data_object(
                    data_object=data_properties,
                    class_name=self._index_name,
                    uuid=_id,
                    vector=embeddings[i] if embeddings else None,
                    tenant=kwargs.get("tenant"),
                )
                ids.append(_id)
        return ids

    def _add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Upload texts with metadata (properties) to Weaviate."""
        from weaviate.util import get_valid_uuid

        ids = []
        embeddings: Optional[List[List[float]]] = None
        if self._embedding:
            if not isinstance(texts, list):
                texts = list(texts)
            embeddings = self._embedding.embed_documents(texts)

        with self._client.batch as batch:
            for i, text in enumerate(texts):
                data_properties = {self._text_key: text}
                if metadatas is not None:
                    for key, val in metadatas[i].items():
                        data_properties[key] = _json_serializable(val)

                _id = get_valid_uuid(uuid4())
                if "uuids" in kwargs:
                    _id = kwargs["uuids"][i]
                elif "ids" in kwargs:
                    _id = kwargs["ids"][i]

                batch.add_data_object(
                    data_object=data_properties,
                    class_name=self._index_name,
                    uuid=_id,
                    vector=embeddings[i] if embeddings else None,
                    tenant=kwargs.get("tenant"),
                )
                ids.append(_id)
        return ids

    def add(self, docs):
        """
               Adds documents to the Weaviate vector store.

               Args:
                   docs: The documents to be added to the vector store.
               """
        try:
            if isinstance(docs[0], Document):
                self._add_documents(documents=docs)
            else:
                self._add_texts(texts=docs)
            logger.info("Document added successfully to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise RuntimeError(f"Can't add document to the vector store: {e}")

    def _similarity_search_with_score(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Return list of documents most similar to the query
        text and cosine distance in float for each.
        Lower score represents more similarity.
        """
        if self._embedding is None:
            raise ValueError(
                "_embedding cannot be None for similarity_search_with_score"
            )
        content: Dict[str, Any] = {"concepts": [query]}
        if kwargs.get("search_distance"):
            content["certainty"] = kwargs.get("search_distance")
        query_obj = self._client.query.get(self._index_name, self._query_attrs)
        if kwargs.get("where_filter"):
            query_obj = query_obj.with_where(kwargs.get("where_filter"))
        if kwargs.get("tenant"):
            query_obj = query_obj.with_tenant(kwargs.get("tenant"))

        embedded_query = self._embedding.embed_query(query)
        # if not self._by_text:
        vector = {"vector": embedded_query}
        result = (
            query_obj.with_near_vector(vector)
            .with_limit(k)
            .with_additional("vector")
            .do()
        )

        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")

        docs_and_scores = []
        for res in result["data"]["Get"][self._index_name]:
            text = res.pop(self._text_key)
            score = np.dot(res["_additional"]["vector"], embedded_query)
            docs_and_scores.append((Document(page_content=text, metadata=res), score))
        return docs_and_scores

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            *,
            client: Optional[weaviate.Client] = None,
            weaviate_url: Optional[str] = None,
            weaviate_api_key: Optional[str] = None,
            batch_size: Optional[int] = None,
            index_name: Optional[str] = None,
            text_key: str = "text",
            by_text: bool = False,
            relevance_score_fn: Optional[
                Callable[[float], float]
            ] = _default_score_normalizer,
            **kwargs: Any,
    ) -> Weaviate:
        """Construct Weaviate wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Weaviate instance.
            3. Adds the documents to the newly created Weaviate index.

        This is intended to be a quick way to get started.

        Args:
            texts: Texts to add to vector store.
            embedding: Text embedding model to use.
            metadatas: Metadata associated with each text.
            client: weaviate.Client to use.
            weaviate_url: The Weaviate URL. If using Weaviate Cloud Services get it
                from the ``Details`` tab. Can be passed in as a named param or by
                setting the environment variable ``WEAVIATE_URL``. Should not be
                specified if client is provided.
            weaviate_api_key: The Weaviate API key. If enabled and using Weaviate Cloud
                Services, get it from ``Details`` tab. Can be passed in as a named param
                or by setting the environment variable ``WEAVIATE_API_KEY``. Should
                not be specified if client is provided.
            batch_size: Size of batch operations.
            index_name: Index name.
            text_key: Key to use for uploading/retrieving text to/from vectorstore.
            by_text: Whether to search by text or by embedding.
            relevance_score_fn: Function for converting whatever distance function the
                vector store uses to a relevance score, which is a normalized similarity
                score (0 means dissimilar, 1 means similar).
            kwargs: Additional named parameters to pass to ``Weaviate.__init__()``.


        """

        try:
            from weaviate.util import get_valid_uuid
        except ImportError as e:
            raise ImportError(
                "Could not import weaviate python  package. "
                "Please install it with `pip install weaviate-client`"
            ) from e

        client = client or _create_weaviate_client(
            url=weaviate_url,
            api_key=weaviate_api_key,
        )
        if batch_size:
            client.batch.configure(batch_size=batch_size)

        index_name = index_name or f"Indox{uuid4().hex}"
        schema = _default_schema(index_name, text_key)
        # check whether the index already exists
        if not client.schema.exists(index_name):
            client.schema.create_class(schema)

        embeddings = embedding.embed_documents(texts) if embedding else None
        attributes = list(metadatas[0].keys()) if metadatas else None

        if "uuids" in kwargs:
            uuids = kwargs.pop("uuids")
        else:
            uuids = [get_valid_uuid(uuid4()) for _ in range(len(texts))]

        with client.batch as batch:
            for i, text in enumerate(texts):
                data_properties = {
                    text_key: text,
                }
                if metadatas is not None:
                    for key in metadatas[i].keys():
                        data_properties[key] = metadatas[i][key]

                _id = uuids[i]

                params = {
                    "uuid": _id,
                    "data_object": data_properties,
                    "class_name": index_name,
                }
                if embeddings is not None:
                    params["vector"] = embeddings[i]

                batch.add_data_object(**params)

            batch.flush()

        return cls(
            client,
            index_name,
            text_key,
            embedding=embedding,
            attributes=attributes,
            relevance_score_fn=relevance_score_fn,
            by_text=by_text,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        for id in ids:
            self._client.data_object.delete(uuid=id)
