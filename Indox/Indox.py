from .Splitter import get_chunks, get_chunks_unstructured
from .Embedding import embedding_model
from .utils import (
     get_user_input, get_metrics, update_config
)
from typing import List, Optional, Any
from .QAModels import choose_qa_model
from .vectorstore import get_vector_store
from .utils import read_config
from .Graph import RAGGraph
import warnings
import tiktoken

warnings.filterwarnings("ignore")


class IndoxRetrievalAugmentation:
    def __init__(self):
        """
        Initialize the IndoxRetrievalAugmentation class

             """
        self.input_tokens_all = 0
        self.embedding_tokens = 0
        self.output_tokens_all = 0
        self.db = None
        self.inputs = {}
        self.unstructured = None
        self.embeddings = None
        self.config = None
        self.qa_model = None
        self.config = read_config()

    def initialize(self):
        """
        Initialize the configuration, embeddings, and QA model.
        Calls `update_config` to update the configuration, then loads
        the embedding model and the QA model.
        """
        # Update the configuration
        update_config(self.config)

        # Initialize embeddings and QA model with the updated configuration
        self.embeddings = embedding_model()
        self.qa_model = choose_qa_model()

    def create_chunks(self, file_path, content_type=None, unstructured=False,
                      max_chunk_size: Optional[int] = 512,
                      re_chunk: bool = False, remove_sword=False):
        """
        Retrieve all chunks from a document based on the provided configurations.

        Parameters:
        - file_path (str): The path to the document file to be processed.
        - content_type (str, optional): The content type of the document, required if `unstructured` is `True`.
        - unstructured (bool, optional): Whether to handle the document as unstructured data. Default is `False`.
        - max_chunk_size (int, optional): The maximum size (in tokens) for each chunk. Default is 512.
        - re_chunk (bool, optional): Whether to re-chunk the structured data. Default is `False`.
        - remove_sword (bool, optional): Whether to exclude specific stopwords during chunking. Default is `False`.

        Returns:
        - list: A list of chunks extracted from the document. Each chunk could be a string or an object, depending on
          the content type and chunking mode.

        Raises:
        - RuntimeError: If both `unstructured` and `re_chunk` are set to `True`.

        Notes:
        - The function supports two modes of document processing: structured and unstructured.
        - It updates class attributes for input, embedding, and output tokens, based on the chunks processed.
        """

        if unstructured and re_chunk:
            raise RuntimeError("Can't re-chunk unstructered data.")
        all_chunks = None
        self.unstructured = unstructured
        embedding_tokens = 0
        try:
            if not unstructured:
                do_clustering = True if get_user_input() == "y" else False
                if do_clustering:
                    all_chunks, input_tokens_all, output_tokens_all = \
                        get_chunks(docs=file_path,
                                   embeddings=self.embeddings,
                                   chunk_size=max_chunk_size,
                                   do_clustering=do_clustering,
                                   re_chunk=re_chunk,
                                   remove_sword=remove_sword)
                    encoding = tiktoken.get_encoding("cl100k_base")
                    embedding_tokens = 0
                    for chunk in all_chunks:
                        token_count = len(encoding.encode(chunk))
                        embedding_tokens = embedding_tokens + token_count
                    self.input_tokens_all = input_tokens_all
                    self.embedding_tokens = embedding_tokens
                    self.output_tokens_all = output_tokens_all
                elif not do_clustering:
                    all_chunks = get_chunks(docs=file_path,
                                            embeddings=self.embeddings,
                                            chunk_size=max_chunk_size,
                                            do_clustering=do_clustering,
                                            re_chunk=re_chunk,
                                            remove_sword=remove_sword)
                    encoding = tiktoken.get_encoding("cl100k_base")
                    embedding_tokens = 0
                    for chunk in all_chunks:
                        token_count = len(encoding.encode(chunk))
                        embedding_tokens = embedding_tokens + token_count
                    self.embedding_tokens = embedding_tokens

            elif unstructured:
                all_chunks = get_chunks_unstructured(file_path=file_path,
                                                     chunk_size=max_chunk_size,
                                                     content_type=content_type,
                                                     remove_sword=remove_sword)
                for chunk in all_chunks:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    token_count = len(encoding.encode(chunk.page_content))
                    embedding_tokens = embedding_tokens + token_count
                self.embedding_tokens = embedding_tokens
            return all_chunks
        except Exception as e:
            print(f"Error while getting chunks: {e}")
            return []

    def connect_to_vectorstore(self, collection_name: str):
        """
        Establish a connection to the vector store database using configuration parameters.

        Parameters:
        - collection_name (str): The name of the collection within the vector store to connect to.

        Raises:
        - RuntimeError: If the connection to the vector store fails or the collection is not found.
        - ValueError: If `collection_name` is not provided or is empty.
        - Exception: Any other error encountered during the connection attempt.

        Returns:
        - None: Prints a message if the connection is successful.
        """
        if not collection_name:
            raise ValueError("Collection name cannot be empty.")

        try:
            self.db = get_vector_store(collection_name=collection_name, embeddings=self.embeddings)

            if self.db is None:
                raise RuntimeError('Failed to connect to the vector store database.')

            print("Connection established successfully.")
        except ValueError as ve:
            raise ValueError(f"Invalid input: {ve}")
        except RuntimeError as re:
            raise RuntimeError(f"Runtime error: {re}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to the database due to an unexpected error: {e}")

    def store_in_vectorstore(self, chunks: List[str]) -> Any:
        """
        Store text chunks into a vector store database.

        Parameters:
        - chunks (List[str]): A list of text chunks to be stored in the vector store.

        Returns:
        - Any: The database object if the chunks are stored successfully; otherwise, `None`.

        Raises:
        - RuntimeError: If the vector store database is not initialized.
        - Exception: Any other error that occurs while storing the chunks.
        """

        if not chunks or not isinstance(chunks, list):
            raise ValueError("The `all_chunks` parameter must be a non-empty list.")

        try:
            if self.db is not None:
                self.db.add_document(chunks, unstructured=self.unstructured)
            else:
                raise RuntimeError("The vector store database is not initialized.")

            return self.db
        except ValueError as ve:
            raise ValueError(f"Invalid input data: {ve}")
        except RuntimeError as re:
            print(f"Runtime error while storing in the vector store: {re}")
            return None
        except Exception as e:
            print(f"Unexpected error while storing in the vector store: {e}")
            return None

    def answer_question(self, query: str, top_k: int = 5, document_relevancy_filter: bool = False):
        """
        Answer a query using the QA model, finding the most relevant document chunks in the database.

        Parameters:
        - query (str): The question or search query to answer.
        - top_k (int): The number of top results to retrieve from the vector store.
        - document_relevancy_filter (bool, optional): If `True`, apply additional filtering using a RAG graph for
          document relevancy. Default is `False`.

        Returns:
        - Tuple[str, Tuple[List[str], List[float]]]: A tuple containing the answer and the retrieved context:
            - `answer` (str): The answer provided by the QA model.
            - `retrieve_context` (Tuple[List[str], List[float]]): A tuple of the retrieved documents and their scores.

        Raises:
        - RuntimeError: If the vector store database is not initialized or fails to retrieve relevant documents.
        - ValueError: If the input query is empty.
        - Exception: Any other unexpected errors that occur while retrieving documents or generating the answer.
        """
        if not query:
            raise ValueError("Query string cannot be empty.")

        if self.db is None:
            raise RuntimeError("Vector store database is not initialized.")

        try:
            context, scores = self.db.retrieve(query, top_k=top_k)
            if not document_relevancy_filter:
                answer = self.qa_model.answer_question(context=context, question=query)
                self.inputs = {"answer": answer, "query": query, "context": context}
            else:
                graph = RAGGraph()
                graph_out = graph.run({'question': query, 'documents': context, 'scores': scores})
                answer = self.qa_model.answer_question(context=graph_out['documents'], question=graph_out['question'])
                context, scores = graph_out['documents'], graph_out['scores']

            retrieve_context = (context, scores)
            return answer, retrieve_context

        except ValueError as ve:
            raise ValueError(f"Invalid input data: {ve}")
        except RuntimeError as re:
            print(f"Runtime error while retrieving or answering the query: {re}")
            return "", []
        except Exception as e:
            print(f"Unexpected error while answering the question: {e}")
            return "", []

    def evaluate(self):
        """
        Evaluate the performance of the system based on the inputs provided from previous queries.

        Returns:
        - dict: The evaluation metrics generated by `get_metrics` if inputs exist.
        - None: If no previous query results are available.

        Raises:
        - RuntimeError: If no previous query results are available to evaluate.
        """
        if self.inputs:
            return get_metrics(self.inputs)
        else:
            raise RuntimeError("No inputs available for evaluation. Please make a query first.")

    def get_tokens_info(self):
        """
        Print an overview of the number of tokens used for different operations.

        Displays the following token counts:
        - `input_tokens_all`: Number of input tokens sent to GPT-3.5 Turbo for summarization.
        - `output_tokens_all`: Number of output tokens received from GPT-3.5 Turbo.
        - `embedding_tokens`: Number of tokens used in the embedding process and sent to the database.

        If no output tokens were used, only the embedding token information is displayed.
        """
        if self.output_tokens_all > 0:
            print(
                f"""
                Overview of All Tokens Used:
                Input tokens sent to GPT-3.5 Turbo (Model ID: 0125) for summarizing: {self.input_tokens_all}
                Output tokens received from GPT-3.5 Turbo (Model ID: 0125): {self.output_tokens_all}
                Tokens used in the embedding section that were sent to the database: {self.embedding_tokens}
                """
            )
        else:
            print(
                f"""
                Overview of All Tokens Used:
                Tokens used in the embedding section that were sent to the database: {self.embedding_tokens}
                """
            )


