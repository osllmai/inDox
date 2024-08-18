import os
from tqdm import tqdm
import json
from pymilvus import MilvusClient
from typing import Optional, Dict, Any, List, Tuple, Type, Callable, Iterable
from indox.core import VectorStore, Embeddings, Document


class Milvus:
    """
    A class for managing a Milvus vector database, including text embeddings,
    document insertion, deletion, and similarity search using SentenceTransformers
    and OpenAI's GPT models.
    """

    def __init__(self, embedding_model: Embeddings, collection_name: str = "indox_collection"):
        """
        Initialize the Milvus client, set the embedding model, and set up
        the collection parameters.

        Parameters
        ----------
        embedding_model : Embeddings
            An instance of a class implementing the Embeddings interface.
        collection_name : str, optional
            The name of the collection to use in Milvus (default is "indox_collection").

        Attributes
        ----------
        embedding_model : Embeddings
            The embedding model used to generate embeddings.
        collection_name : str
            The name of the collection to use in Milvus.
        """
        self.milvus_client = MilvusClient(host='127.0.0.1', port='19530')
        self.collection_name = collection_name
        self.embedding_dim = None
        self.embedding_model = embedding_model

    def load_text_lines(self, file_path: str) -> List[str]:
        """
        Load text lines from a .txt file.

        Parameters
        ----------
        file_path : str
            The path to the text file.

        Returns
        -------
        List[str]
            A list of lines loaded from the text file.
        """
        with open(file_path, 'r') as file:
            text_lines = file.readlines()
        text_lines = [line.strip() for line in text_lines]
        return text_lines

    def emb_text(self, text: str) -> List[float]:
        """
        Generate the embedding for a given text using the provided embedding model.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        List[float]
            The generated embedding as a list of floats.
        """
        embedding = self.embedding_model.embed_query(text)
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        return embedding

    def create_collection(self) -> None:
        """
        Create a collection in Milvus with the appropriate settings.

        Raises
        ------
        ValueError
            If the embedding dimension is not set.
        """
        if self.embedding_dim is None:
            raise ValueError("Embedding dimension is not set. Ensure that you generate an embedding first.")

        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="IP",
            consistency_level="Strong",
        )

    def insert_data(self, text_lines: List[str]) -> None:
        """
        Insert a list of text lines into the Milvus collection.

        Parameters
        ----------
        text_lines : List[str]
            The list of text lines to insert.
        """
        data = [{"id": i, "vector": self.emb_text(line), "text": line} for i, line in
                enumerate(tqdm(text_lines, desc="Creating embeddings"))]
        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def similarity_search_with_score(self, question: str, limit: int = 3) -> List[tuple]:
        """
        Perform a similarity search in the Milvus collection for a given question.

        Parameters
        ----------
        question : str
            The question to search for.
        limit : int, optional
            The maximum number of results to return (default is 3).

        Returns
        -------
        List[tuple]
            A list of tuples, each containing a retrieved text and its similarity score.
        """
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[self.emb_text(question)],
            limit=limit,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
        return [
            (res["entity"]["text"], res["distance"])
            for res in search_res[0]
        ]

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate an answer to a question using the provided context with OpenAI's GPT model.

        Parameters
        ----------
        context : str
            The context from which to generate the answer.
        question : str
            The question to be answered.

        Returns
        -------
        str
            The generated answer.
        """
        SYSTEM_PROMPT = """
        Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
        """
        USER_PROMPT = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )
        return response.choices[0].message['content']

    def process_question(self, question: str) -> None:
        """
        Process a question by performing a similarity search and generating an answer.

        Parameters
        ----------
        question : str
            The question to be processed.
        """
        retrieved_lines_with_distances = self.similarity_search_with_score(question)
        context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
        answer = self.generate_answer(context, question)
        print(json.dumps(retrieved_lines_with_distances, indent=4))
        print(answer)

    def add_documents(self, documents: List[str]) -> List[str]:
        """
        Add more documents to the vector store by generating embeddings and inserting them.

        Parameters
        ----------
        documents : List[str]
            The documents to add to the vector store.

        Returns
        -------
        List[str]
            A list of IDs of the added texts.
        """
        new_ids = []
        for i, doc in enumerate(documents):
            embedding = self.emb_text(doc)
            doc_id = str(len(new_ids))  # Generate a new ID
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=[{"id": doc_id, "vector": embedding, "text": doc}]
            )
            new_ids.append(doc_id)
        return new_ids

    def delete_collection(self) -> None:
        """
        Delete the entire collection in Milvus.
        """
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

    def update_document(self, document_id: str, document: str) -> None:
        """
        Update an existing document in the Milvus collection.

        Parameters
        ----------
        document_id : str
            The ID of the document to update.
        document : str
            The new content for the document.
        """
        embedding = self.emb_text(document)
        self.milvus_client.update(
            collection_name=self.collection_name,
            data=[{"id": document_id, "vector": embedding, "text": document}]
        )

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete documents from the Milvus collection by their IDs.

        Parameters
        ----------
        ids : List[str], optional
            The IDs of the documents to delete (default is None).
        """
        if ids is not None:
            self.milvus_client.delete(
                collection_name=self.collection_name,
                ids=ids
            )

    def __len__(self) -> int:
        """
        Count the number of documents in the Milvus collection.

        Returns
        -------
        int
            The number of documents in the collection.
        """
        return self.milvus_client.count(collection_name=self.collection_name)
