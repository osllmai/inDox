from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Type, Dict

import numpy as np
from indox.core import Document, Embeddings, VectorStore


class Atlas:
    """`Atlas` vector store.

     Atlas is the `Nomic's` neural database and `rhizomatic` instrument.

    To use, you should have the ``nomic`` python package installed.

    """

    _ATLAS_DEFAULT_ID_FIELD: str = "atlas_id"

    def __init__(
            self,
            name: str,
            embedding_function: Optional[Embeddings] = None,
            api_key: Optional[str] = None,
            description: str = "A description for your project",
            is_public: bool = True,
            reset_project_if_exists: bool = False,
    ) -> None:
        """
        Initialize the Atlas Client

        Args:
            name (str): The name of your project. If the project already exists,
                it will be loaded.
            embedding_function (Optional[Embeddings]): An optional function used for
                embedding your data. If None, data will be embedded with
                Nomic's embed model.
            api_key (str): Your nomic API key
            description (str): A description for your project.
            is_public (bool): Whether your project is publicly accessible.
                True by default.
            reset_project_if_exists (bool): Whether to reset this project if it
                already exists. Default False.
                Generally useful during development and testing.
        """
        try:
            import nomic
            from nomic import AtlasDataset
        except ImportError:
            raise ImportError(
                "Could not import nomic python package. "
                "Please install it with `pip install nomic`."
            )

        if api_key is None:
            raise ValueError("No API key provided. Sign up at atlas.nomic.ai!")
        nomic.login(api_key)

        self._embedding_function = embedding_function
        modality = "text"
        if self._embedding_function is not None:
            modality = "embedding"

        # Check if the project exists, create it if not
        self.project = AtlasDataset(
            identifier=name,
            description=description,
            # modality=modality,
            is_public=is_public,
            # reset_project_if_exists=reset_project_if_exists,
            # unique_id_field=Atlas._ATLAS_DEFAULT_ID_FIELD,
        )
        self.project._latest_dataset_state()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    def add_documents(
            self,
            documents: Iterable[Dict[str, Any]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            refresh: bool = True,
            **kwargs: Any,
    ) -> List[str]:
        """
        Run more documents through the embeddings and add them to the vectorstore.

        Args:
            documents (Iterable[Dict[str, Any]]): Documents to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]]): An optional list of ids.
            refresh (bool): Whether or not to refresh indices with the updated data.
                Default True.
        Returns:
            List[str]: List of IDs of the added documents.
        """

        if (
                metadatas is not None
                and len(metadatas) > 0
                and "text" in metadatas[0].keys()
        ):
            raise ValueError("Cannot accept key 'text' in metadata!")

        documents = list(documents)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Embedding upload case
        if self._embedding_function is not None:
            texts = [doc["text"] for doc in documents]
            _embeddings = self._embedding_function.embed_documents(texts)
            embeddings = np.stack(_embeddings)

            if metadatas is None:
                data = [
                    {**documents[i], Atlas._ATLAS_DEFAULT_ID_FIELD: ids[i]}
                    for i in range(len(documents))
                ]
            else:
                for i in range(len(metadatas)):
                    metadatas[i][Atlas._ATLAS_DEFAULT_ID_FIELD] = ids[i]
                    metadatas[i].update(documents[i])
                data = metadatas

            self.project._validate_map_data_inputs(
                [], id_field=Atlas._ATLAS_DEFAULT_ID_FIELD, data_sample=data
            )
            with self.project.wait_for_dataset_lock():
                self.project._add_embeddings(embeddings=embeddings, data=data)

        # Document upload case without embeddings
        else:
            if metadatas is None:
                data = [
                    {**documents[i], Atlas._ATLAS_DEFAULT_ID_FIELD: ids[i]}
                    for i in range(len(documents))
                ]
            else:
                for i, doc in enumerate(documents):
                    metadatas[i][Atlas._ATLAS_DEFAULT_ID_FIELD] = ids[i]
                    metadatas[i].update(doc)
                data = metadatas

            self.project._validate_map_data_inputs(
                [], id_field=Atlas._ATLAS_DEFAULT_ID_FIELD, data_sample=data
            )

            with self.project.wait_for_dataset_lock():
                self.project._add_text(data)

        if refresh:
            if len(self.project.indices) > 0:
                with self.project.wait_for_dataset_lock():
                    self.project.update_indices()

        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            refresh: bool = True,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]]): An optional list of ids.
            refresh(bool): Whether or not to refresh indices with the updated data.
                Default True.
        Returns:
            List[str]: List of IDs of the added texts.
        """

        if (
                metadatas is not None
                and len(metadatas) > 0
                and "text" in metadatas[0].keys()
        ):
            raise ValueError("Cannot accept key text in metadata!")

        texts = list(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Embedding upload case
        if self._embedding_function is not None:
            _embeddings = self._embedding_function.embed_documents(texts)
            embeddings = np.stack(_embeddings)
            if metadatas is None:
                data = [
                    {Atlas._ATLAS_DEFAULT_ID_FIELD: ids[i], "text": texts[i]}
                    for i, _ in enumerate(texts)
                ]
            else:
                for i in range(len(metadatas)):
                    metadatas[i][Atlas._ATLAS_DEFAULT_ID_FIELD] = ids[i]
                    metadatas[i]["text"] = texts[i]
                data = metadatas

            self.project._validate_map_data_inputs(
                [], id_field=Atlas._ATLAS_DEFAULT_ID_FIELD, data_sample=data
            )
            with self.project.wait_for_dataset_lock():
                self.project._add_embeddings(embeddings=embeddings, data=data)
        # Text upload case
        else:
            if metadatas is None:
                data = [
                    {"text": text, Atlas._ATLAS_DEFAULT_ID_FIELD: ids[i]}
                    for i, text in enumerate(texts)
                ]
            else:
                for i, text in enumerate(texts):
                    metadatas[i]["text"] = texts
                    metadatas[i][Atlas._ATLAS_DEFAULT_ID_FIELD] = ids[i]
                data = metadatas

            self.project._validate_map_data_inputs(
                [], id_field=Atlas._ATLAS_DEFAULT_ID_FIELD, data_sample=data
            )

            with self.project.wait_for_dataset_lock():
                self.project._add_text(data)

        if refresh:
            if len(self.project.indices) > 0:
                with self.project.wait_for_dataset_lock():
                    self.project.update_indices()

        return ids

    def create_index(self, **kwargs: Any) -> Any:
        """Creates an index in your project.

        See
        https://docs.nomic.ai/atlas_api.html#nomic.project.AtlasProject.create_index
        for full detail.
        """
        with self.project.wait_for_dataset_lock():
            return self.project.create_index(**kwargs)

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with AtlasDB

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        if self._embedding_function is None:
            raise NotImplementedError(
                "AtlasDB requires an embedding_function for text similarity search!"
            )
        _embedding = self._embedding_function.embed_documents([query])[0]
        embedding = np.array(_embedding).reshape(1, -1)


        with self.project.wait_for_dataset_lock():
            print(self.project.projections[0])
            neighbors, _ = self.project.projections[0].vector_search(
                queries=embedding, k=k
            )
            if not neighbors:
                raise ValueError("No neighbors found for the given query.")

            data = self.project.get_data(ids=neighbors[0])

            if not data:
                raise ValueError("No data found for the retrieved neighbors.")

        docs = [
            Document(page_content=data[i]["text"], metadata=data[i])
            for i, neighbor in enumerate(neighbors)
        ]
        return docs
    @classmethod
    def from_texts(
            cls: Type[Atlas],
            texts: List[str],
            embedding: Optional[Embeddings] = None,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            name: Optional[str] = None,
            api_key: Optional[str] = None,
            description: str = "A description for your project",
            is_public: bool = True,
            reset_project_if_exists: bool = False,
            index_kwargs: Optional[dict] = None,
            **kwargs: Any,
    ) -> Atlas:
        """Create an AtlasDB vectorstore from a raw documents.

        Args:
            texts (List[str]): The list of texts to ingest.
            name (str): Name of the project to create.
            api_key (str): Your nomic API key,
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): Optional list of document IDs. If None,
                ids will be auto created
            description (str): A description for your project.
            is_public (bool): Whether your project is publicly accessible.
                True by default.
            reset_project_if_exists (bool): Whether to reset this project if it
                already exists. Default False.
                Generally useful during development and testing.
            index_kwargs (Optional[dict]): Dict of kwargs for index creation.
                See https://docs.nomic.ai/atlas_api.html

        Returns:
            AtlasDB: Nomic's neural database and finest rhizomatic instrument
        """
        if name is None or api_key is None:
            raise ValueError("`name` and `api_key` cannot be None.")

        # Inject relevant kwargs
        all_index_kwargs = {"name": name + "_index", "indexed_field": "text"}
        if index_kwargs is not None:
            for k, v in index_kwargs.items():
                all_index_kwargs[k] = v

        # Build project
        atlasDB = cls(
            name,
            embedding_function=embedding,
            api_key=api_key,
            description="A description for your project",
            is_public=is_public,
            reset_project_if_exists=reset_project_if_exists,
        )
        with atlasDB.project.wait_for_dataset_lock():
            atlasDB.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            atlasDB.create_index(**all_index_kwargs)
        return atlasDB

    @classmethod
    def from_documents(
            cls: Type[Atlas],
            documents: List[Document],
            embedding: Optional[Embeddings] = None,
            ids: Optional[List[str]] = None,
            name: Optional[str] = None,
            api_key: Optional[str] = None,
            persist_directory: Optional[str] = None,
            description: str = "A description for your project",
            is_public: bool = True,
            reset_project_if_exists: bool = False,
            index_kwargs: Optional[dict] = None,
            **kwargs: Any,
    ) -> Atlas:
        """Create an AtlasDB vectorstore from a list of documents.

        Args:
            name (str): Name of the collection to create.
            api_key (str): Your nomic API key,
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            ids (Optional[List[str]]): Optional list of document IDs. If None,
                ids will be auto created
            description (str): A description for your project.
            is_public (bool): Whether your project is publicly accessible.
                True by default.
            reset_project_if_exists (bool): Whether to reset this project if
                it already exists. Default False.
                Generally useful during development and testing.
            index_kwargs (Optional[dict]): Dict of kwargs for index creation.
                See https://docs.nomic.ai/atlas_api.html

        Returns:
            AtlasDB: Nomic's neural database and finest rhizomatic instrument
        """
        if name is None or api_key is None:
            raise ValueError("`name` and `api_key` cannot be None.")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            name=name,
            api_key=api_key,
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            description=description,
            is_public=is_public,
            reset_project_if_exists=reset_project_if_exists,
            index_kwargs=index_kwargs,
        )
