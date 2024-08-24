import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Literal, Iterable
import uuid
from indox.core import Embeddings, VectorStore, Document
from loguru import logger
import sys


# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")



class Milvus:
    """
    A wrapper class for interacting with the Milvus vector database.
    """
    def __init__(self, collection_name, embedding_model):
        """
        Initialize the MilvusWrapper with collection name, embedding model, and QA model.

        Args:
            collection_name (str): The name of the collection in Milvus.
            embedding_model (object): An object with methods `embed_query` and `embed_documents`.
        """
        from pymilvus import MilvusClient, MilvusException

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = None
        self.milvus_client = MilvusClient(host='127.0.0.1', port='19530')


    def _embed_query(self, query: str) -> List[float]:
        """
        Embed the query into a vector.

        Args:
            query (str): The query text to embed.

        Returns:
            List[float]: The embedding vector for the query.
        """
        return self.embedding_model.embed_query(query)

    def _similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Return docs most similar to the query.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): Number of Documents to return. Defaults to 3.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[self.emb_text(query)],
            limit=k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )

        # Creating Document objects for the results
        documents_with_scores = [
            (Document(page_content=res["entity"]["text"]), res["distance"])
            for res in search_res[0]
        ]
        return documents_with_scores
    def _similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Document]: Document results for search.
        """
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        res = self._similarity_search_with_score(
            query=query, k=k
        )
        return [doc for doc, _ in res]

    def _process_question(self, question: str):
        """
        Process a question by retrieving relevant documents and printing them.

        Args:
            question (str): The question to process.
        """
        retrieved_lines_with_distances = self._similarity_search_with_score(question, k=5)
        # Convert Document objects to dictionaries
        context = "\n".join([self._to_dict(doc)['page_content'] for doc, _ in retrieved_lines_with_distances])
        # answer = self.generate_answer(context, question)
        # print(f"Answer: {answer}")
        print(json.dumps(
            [{"document": self._to_dict(doc), "score": score} for doc, score in retrieved_lines_with_distances],
            indent=4
        ))

    # def store_in_vectorstore(self, docs: List[Document]):
    #     """
    #     Store documents in the Milvus vector store.
    #
    #     Args:
    #         docs (List[Document]): List of Document objects to store.
    #     """
    #     text_lines = [doc.page_content for doc in docs]
    #     self.insert_data(text_lines)

    def emb_text(self, text: str) -> List[float]:
        """
        Get the embedding vector for the provided text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector for the text.
        """
        embedding = self.embedding_model.embed_documents([text])[0]
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        return embedding

    def load_text_from_file(self, file_path: str) -> List[str]:
        """
        Load text lines from a file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            List[str]: List of text lines read from the file.
        """
        with open(file_path, "r") as file:
            text_lines = file.readlines()
        return text_lines

    def insert_data(self, text_lines: List[str]):
        """
        Insert data into the Milvus vector store.

        Args:
            text_lines (List[str]): List of text lines to insert.
        """
        from tqdm import tqdm

        data = [{"id": i, "vector": self.emb_text(line), "text": line} for i, line in
                enumerate(tqdm(text_lines, desc="Creating embeddings"))]
        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def _add(
            self,
            texts: Iterable[str],
            embeddings: Iterable[List[float]],
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents with embeddings to the vector store.

        Args:
            texts (Iterable[str]): Texts to add.
            embeddings (Iterable[List[float]]): Corresponding embeddings for the texts.
            metadatas (Optional[Iterable[dict]]): Optional metadata for the documents.
            ids (Optional[List[str]]): Optional IDs for the documents.

        Returns:
            List[str]: List of document IDs.
        """
        _metadatas = metadatas or ({} for _ in texts)
        documents = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
        ]

        if ids and len(ids) != len(set(ids)):
            raise ValueError("Duplicate ids found in the ids list.")

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for id_, doc, embedding in zip(ids, documents, embeddings):
            self.milvus_client.add(embedding=embedding, metadata=doc.metadata)
        return ids

    def add(self, docs):
        """
        Adds documents to the Milvus vector store.

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

    def _add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """
        Add texts with embeddings to the vector store.

        Args:
            texts (Iterable[str]): Texts to add.
            metadatas (Optional[List[dict]]): Optional metadata for the documents.
            ids (Optional[List[str]]): Optional IDs for the documents.

        Returns:
            List[str]: List of document IDs.
        """
        texts = list(texts)
        embeddings = self.embedding_model.embed_documents(texts)
        return self._add(texts, embeddings, metadatas=metadatas, ids=ids)

    def add_embeddings(
            self,
            text_embeddings: Iterable[Tuple[str, List[float]]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """
        Add embeddings with corresponding texts to the vector store.

        Args:
            text_embeddings (Iterable[Tuple[str, List[float]]]): Iterable of text and embedding pairs.
            metadatas (Optional[List[dict]]): Optional metadata for the documents.
            ids (Optional[List[str]]): Optional IDs for the documents.

        Returns:
            List[str]: List of document IDs.
        """
        texts, embeddings = zip(*text_embeddings)
        return self.add(texts, embeddings, metadatas=metadatas, ids=ids)

    def _add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]: Documents to add to the vectorstore.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self._add_texts(texts, metadatas, **kwargs)

    def _to_dict(self, document: Document) -> Dict[str, Any]:
        """
        Convert a Document to a dictionary.

        Args:
            document (Document): Document object to convert.

        Returns:
            dict: Dictionary containing the page content and metadata.
        """
        return {
            "page_content": document.page_content,
            "metadata": document.metadata
        }
