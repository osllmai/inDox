from .utils import update_config
from typing import List, Optional, Any
from .vectorstore import get_vector_store
from .utils import read_config
from .Graph import RAGGraph
import warnings

from .visualization import visualize_contexts_

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
        self.config = None
        self.qa_model = None
        self.config = read_config()
        self.test = None
        self.qa_history = []

    def update_config(self):
        """
        Calls `update_config` to update the configuration, then loads
        """
        # Update the configuration
        update_config(self.config)

    def connect_to_vectorstore(self, embeddings, collection_name: str):
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
            self.db = get_vector_store(collection_name=collection_name, embeddings=embeddings)

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
                self.db.add_document(chunks)
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

    def answer_question(self, qa_model, query: str, top_k: int = 5, document_relevancy_filter: bool = False):
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
                # TODO: Add cost of embedding and qa_model
                # response = qa_model.answer_question(context=context, question=query)
                # self.output_tokens += response.usage.completion_tokens
                # self.input_tokens += response.usage.prompt_tokens
                answer = qa_model.answer_question(context=context, question=query)
            else:
                graph = RAGGraph()
                graph_out = graph.run({'question': query, 'documents': context, 'scores': scores})
                answer = qa_model.answer_question(context=graph_out['documents'], question=graph_out['question'])
                context, scores = graph_out['documents'], graph_out['scores']
            retrieve_context = (context, scores)
            new_entry = {'query': query, 'answer': answer, 'context': context, 'scores': scores}
            self.qa_history.append(new_entry)
            return answer, retrieve_context

        except ValueError as ve:
            raise ValueError(f"Invalid input data: {ve}")
        except RuntimeError as re:
            print(f"Runtime error while retrieving or answering the query: {re}")
            return "", []
        except Exception as e:
            print(f"Unexpected error while answering the question: {e}")
            return "", []

    # TODO add visualization for evaluation
    # def evaluate(self):
    #     """
    #     Evaluate the performance of the system based on the inputs provided from previous queries.
    #
    #     Returns:
    #     - dict: The evaluation metrics generated by `get_metrics` if inputs exist.
    #     - None: If no previous query results are available.
    #
    #     Raises:
    #     - RuntimeError: If no previous query results are available to evaluate.
    #     """
    #     if self.inputs:
    #         return get_metrics(self.inputs)
    #     else:
    #         raise RuntimeError("No inputs available for evaluation. Please make a query first.")

    def visualize_context(self):
        """
        Visualize the context of the last query made by the user.
        """
        if not self.qa_history:
            print("No entries to visualize.")
            return

        last_entry = self.qa_history[-1]
        return visualize_contexts_(last_entry['query'], last_entry['context'], last_entry['scores'])

    # def get_tokens_info(self):
    #     """
    #     Print an overview of the number of tokens used for different operations.
    #
    #     Displays the following token counts:
    #     - `input_tokens_all`: Number of input tokens sent to GPT-3.5 Turbo for summarization.
    #     - `output_tokens_all`: Number of output tokens received from GPT-3.5 Turbo.
    #     - `embedding_tokens`: Number of tokens used in the embedding process and sent to the database.
    #
    #     If no output tokens were used, only the embedding token information is displayed.
    #     """
        # if self.output_tokens_all > 0:
        #     print(
        #         f"""
        #         Overview of All Tokens Used:
        #         Input tokens sent to GPT-3.5 Turbo (Model ID: 0125) for summarizing: {self.input_tokens_all}
        #         Output tokens received from GPT-3.5 Turbo (Model ID: 0125): {self.output_tokens_all}
        #         Tokens used in the embedding section that were sent to the database: {self.embedding_tokens}
        #         """
        #     )
        # else:
        #     print(
        #         f"""
        #         Overview of All Tokens Used:
        #         Tokens used in the embedding section that were sent to the database: {self.embedding_tokens}
        #         """
        #     )
