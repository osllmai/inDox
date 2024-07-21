from __future__ import annotations
import logging
import os
import pickle
import operator
import uuid
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sized,
    Tuple,
    Union,
)
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from indox.core import Embeddings, VectorStore, Document


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


class Docstore(ABC):
    """Interface to access to place that stores documents."""

    @abstractmethod
    def search(self, search: str) -> Union[str, Document]:
        """Search for document.

        If page exists, return the page summary, and a Document object.
        If page does not exist, return similar entries.
        """

    def delete(self, ids: List) -> None:
        """Deleting IDs from in memory dictionary."""
        raise NotImplementedError


class AddableMixin(ABC):
    """Mixin class that supports adding texts."""

    @abstractmethod
    def add(self, texts: Dict[str, Document]) -> None:
        """Add more documents."""


logger = logging.getLogger(__name__)


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


class InMemoryDocstore(Docstore, AddableMixin):
    """Simple in memory docstore in the form of a dict."""

    def __init__(self, _dict: Optional[Dict[str, Document]] = None):
        """Initialize with dict."""
        self._dict = _dict if _dict is not None else {}

    def add(self, texts: Dict[str, Document]) -> None:
        """Add texts to in memory dictionary.

        Args:
            texts: dictionary of id -> document.

        Returns:
            None
        """
        overlapping = set(texts).intersection(self._dict)
        if overlapping:
            raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        self._dict = {**self._dict, **texts}

    def delete(self, ids: List) -> None:
        """Deleting IDs from in memory dictionary."""
        overlapping = set(ids).intersection(self._dict)
        if not overlapping:
            raise ValueError(f"Tried to delete ids that does not  exist: {ids}")
        for _id in ids:
            self._dict.pop(_id)

    def search(self, search: str) -> Union[str, Document]:
        """Search via direct lookup.

        Args:
            search: id of a document to search for.

        Returns:
            Document if found, else error message.
        """
        if search not in self._dict:
            return f"ID {search} not found."
        else:
            return self._dict[search]


def _len_check_if_sized(x: Any, y: Any, x_name: str, y_name: str) -> None:
    if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
        raise ValueError(
            f"{x_name} and {y_name} expected to be equal length but "
            f"len({x_name})={len(x)} and len({y_name})={len(y)}"
        )
    return


class FAISS(ABC):
    """`Meta Faiss` vector store.

    To use, you must have the ``faiss`` python package installed.
    """

    def __init__(
            self,
            embedding_function: Union[
                Callable[[str], List[float]],
                Embeddings,
            ],
            relevance_score_fn: Optional[Callable[[float], float]] = None,
            normalize_L2: bool = False,
            distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        """Initialize with necessary components."""
        import faiss
        if not isinstance(embedding_function, Embeddings):
            logger.warning(
                "`embedding_function` is expected to be an Embeddings object, support "
                "for passing in a function will soon be removed."
            )
        self.embedding_function = embedding_function
        embedding_dim = len(self.embedding_function.embed_query(""))

        self.index = faiss.IndexFlatL2(embedding_dim)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        if (
                self.distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE
                and self._normalize_L2
        ):
            warnings.warn(
                "Normalizing L2 is not applicable for "
                f"metric type: {self.distance_strategy}"
            )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_documents(texts)
        else:
            return [self.embedding_function(text) for text in texts]

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_query(text)
        else:
            return self.embedding_function(text)

    def __add(
            self,
            texts: Iterable[str],
            embeddings: Iterable[List[float]],
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
    ) -> List[str]:
        faiss = dependable_faiss_import()

        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )

        _len_check_if_sized(texts, metadatas, "texts", "metadatas")
        _metadatas = metadatas or ({} for _ in texts)
        documents = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
        ]

        _len_check_if_sized(documents, embeddings, "documents", "embeddings")
        _len_check_if_sized(documents, ids, "documents", "ids")

        if ids and len(ids) != len(set(ids)):
            raise ValueError("Duplicate ids found in the ids list.")

        # Add to the index.
        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        self.index.add(vector)

        # Add information to docstore and index.
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        self.index_to_docstore_id.update(index_to_id)
        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        embeddings = self._embed_documents(texts)
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids)

    def add_embeddings(
            self,
            text_embeddings: Iterable[Tuple[str, List[float]]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Add the given texts and embeddings to the vectorstore.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # Embed and create the documents.
        texts, embeddings = zip(*text_embeddings)
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids)

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[Union[Callable, Dict[str, Any]]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.

            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """
        embedding = self._embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[Union[Callable, Dict[str, Any]]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Union[Callable, Dict[str, Any]]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        docs = []

        if filter is not None:
            filter_func = self._create_filter_func(filter)

        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            if filter is not None:
                if filter_func(doc.metadata):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            cmp = (
                operator.ge
                if self.distance_strategy
                   in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                else operator.le
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]
        return docs[:k]
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")
        missing_ids = set(ids).difference(self.index_to_docstore_id.values())
        if missing_ids:
            raise ValueError(
                f"Some specified ids do not exist in the current store. Ids not found: "
                f"{missing_ids}"
            )

        reversed_index = {id_: idx for idx, id_ in self.index_to_docstore_id.items()}
        index_to_delete = {reversed_index[id_] for id_ in ids}

        self.index.remove_ids(np.fromiter(index_to_delete, dtype=np.int64))
        self.docstore.delete(ids)

        remaining_ids = [
            id_
            for i, id_ in sorted(self.index_to_docstore_id.items())
            if i not in index_to_delete
        ]
        self.index_to_docstore_id = {i: id_ for i, id_ in enumerate(remaining_ids)}

        return True

    def merge_from(self, target: FAISS) -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Get id and docs from target FAISS object
        full_info = []
        for i, target_id in target.index_to_docstore_id.items():
            doc = target.docstore.search(target_id)
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, target_id, doc))

        # Add information to docstore and index_to_docstore_id.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)

    @classmethod
    def __from(
            cls,
            texts: Iterable[str],
            embeddings: List[List[float]],
            embedding: Embeddings,
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
            normalize_L2: bool = False,
            distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
            **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(embeddings[0]))
        docstore = kwargs.pop("docstore", InMemoryDocstore())
        index_to_docstore_id = kwargs.pop("index_to_docstore_id", {})
        vecstore = cls(
            embedding,
            index,
            docstore,
            index_to_docstore_id,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )
        vecstore.__add(texts, embeddings, metadatas=metadatas, ids=ids)
        return vecstore

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.


        """
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
            cls,
            texts: list[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents asynchronously.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.
        """
        embeddings = await embedding.aembed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
            cls,
            text_embeddings: Iterable[Tuple[str, List[float]]],
            embedding: Embeddings,
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database
        """
        texts, embeddings = zip(*text_embeddings)
        return cls.__from(
            list(texts),
            list(embeddings),
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_embeddings(
            cls,
            text_embeddings: Iterable[Tuple[str, List[float]]],
            embedding: Embeddings,
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents asynchronously."""
        return cls.from_embeddings(
            text_embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.index, str(path / f"{index_name}.faiss"))

        # save docstore and index_to_docstore_id
        with open(path / f"{index_name}.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
            cls,
            folder_path: str,
            embeddings: Embeddings,
            index_name: str = "index",
            *,
            allow_dangerous_deserialization: bool = False,
            **kwargs: Any,
    ) -> FAISS:
        """Load FAISS index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
            allow_dangerous_deserialization: whether to allow deserialization
                of the data which involves loading a pickle file.
                Pickle files can be modified by malicious actors to deliver a
                malicious payload that results in execution of
                arbitrary code on your machine.
        """
        if not allow_dangerous_deserialization:
            raise ValueError(
                "The de-serialization relies loading a pickle file. "
                "Pickle files can be modified to deliver a malicious payload that "
                "results in execution of arbitrary code on your machine."
                "You will need to set `allow_dangerous_deserialization` to `True` to "
                "enable deserialization. If you do this, make sure that you "
                "trust the source of the data. For example, if you are loading a "
                "file that you created, and know that no one else has modified the "
                "file, then this is safe to do. Do not set this to `True` if you are "
                "loading a file from an untrusted source (e.g., some random site on "
                "the internet.)."
            )
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / f"{index_name}.faiss"))

        # load docstore and index_to_docstore_id
        with open(path / f"{index_name}.pkl", "rb") as f:
            (
                docstore,
                index_to_docstore_id,
            ) = pickle.load(  # ignore[pickle]: explicit-opt-in
                f
            )

        return cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)

    def serialize_to_bytes(self) -> bytes:
        """Serialize FAISS index, docstore, and index_to_docstore_id to bytes."""
        return pickle.dumps((self.index, self.docstore, self.index_to_docstore_id))

    @classmethod
    def deserialize_from_bytes(
            cls,
            serialized: bytes,
            embeddings: Embeddings,
            *,
            allow_dangerous_deserialization: bool = False,
            **kwargs: Any,
    ) -> FAISS:
        """Deserialize FAISS index, docstore, and index_to_docstore_id from bytes."""
        if not allow_dangerous_deserialization:
            raise ValueError(
                "The de-serialization relies loading a pickle file. "
                "Pickle files can be modified to deliver a malicious payload that "
                "results in execution of arbitrary code on your machine."
                "You will need to set `allow_dangerous_deserialization` to `True` to "
                "enable deserialization. If you do this, make sure that you "
                "trust the source of the data. For example, if you are loading a "
                "file that you created, and know that no one else has modified the "
                "file, then this is safe to do. Do not set this to `True` if you are "
                "loading a file from an untrusted source (e.g., some random site on "
                "the internet.)."
            )
        (
            index,
            docstore,
            index_to_docstore_id,
        ) = pickle.loads(  # ignore[pickle]: explicit-opt-in
            serialized
        )
        return cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided in
        # vectorstore constructor
        if self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            # Default behavior is to use euclidean distance relevancy
            return self._euclidean_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be cosine, max_inner_product,"
                " or euclidean"
            )

    @staticmethod
    def _create_filter_func(
            filter: Optional[Union[Callable, Dict[str, Any]]],
    ) -> Callable[[Dict[str, Any]], bool]:
        """
        Create a filter function based on the provided filter.

        Args:
            filter: A callable or a dictionary representing the filter
            conditions for documents.

        Returns:
            Callable[[Dict[str, Any]], bool]: A function that takes Document's metadata
            and returns True if it satisfies the filter conditions, otherwise False.
        """
        if callable(filter):
            return filter

        if not isinstance(filter, dict):
            raise ValueError(
                f"filter must be a dict of metadata or a callable, not {type(filter)}"
            )

        def filter_func(metadata: Dict[str, Any]) -> bool:
            return all(
                metadata.get(key) in value
                if isinstance(value, list)
                else metadata.get(key) == value
                for key, value in filter.items()  # type: ignore
            )

        return filter_func
