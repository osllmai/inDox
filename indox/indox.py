from typing import List, Any, Tuple
import warnings
import logging
from .utils import show_indox_logo

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class IndoxRetrievalAugmentation:
    def __init__(self):
        """
        Initialize the IndoxRetrievalAugmentation class
        """
        from . import __version__
        self.__version__ = __version__
        self.db = None
        self.qa_history = []
        logging.info("IndoxRetrievalAugmentation initialized")
        show_indox_logo()

    def connect_to_vectorstore(self, vectorstore_database):
        """
        Establish a connection to the vector store database using configuration parameters.
        """
        try:
            logging.info("Attempting to connect to the vector store database")
            self.db = vectorstore_database

            if self.db is None:
                raise RuntimeError('Failed to connect to the vector store database.')

            logging.info("Connection to the vector store database established successfully")
            return self.db
        except ValueError as ve:
            logging.error(f"Invalid input: {ve}")
            raise ValueError(f"Invalid input: {ve}")
        except RuntimeError as re:
            logging.error(f"Runtime error: {re}")
            raise RuntimeError(f"Runtime error: {re}")
        except Exception as e:
            logging.error(f"Failed to connect to the database due to an unexpected error: {e}")
            raise RuntimeError(f"Failed to connect to the database due to an unexpected error: {e}")

    def store_in_vectorstore(self, docs: List[str]) -> Any:
        """
        Store text chunks into a vector store database.
        """
        if not docs or not isinstance(docs, list):
            logging.error("The `docs` parameter must be a non-empty list.")
            raise ValueError("The `docs` parameter must be a non-empty list.")

        try:
            logging.info("Storing documents in the vector store")
            if self.db is not None:
                self.db.add_document(docs)
            else:
                raise RuntimeError("The vector store database is not initialized.")

            logging.info("Documents stored successfully")
            return self.db
        except ValueError as ve:
            logging.error(f"Invalid input data: {ve}")
            raise ValueError(f"Invalid input data: {ve}")
        except RuntimeError as re:
            logging.error(f"Runtime error while storing in the vector store: {re}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error while storing in the vector store: {e}")
            return None

    class QuestionAnswer:
        def __init__(self, llm, vector_database, top_k: int = 5, document_relevancy_filter: bool = False,
                     generate_clustered_prompts: bool = False):
            self.qa_model = llm
            self.document_relevancy_filter = document_relevancy_filter
            self.top_k = top_k
            self.generate_clustered_prompts = generate_clustered_prompts
            self.vector_database = vector_database
            self.qa_history = []
            self.context = []
            if self.vector_database is None:
                logging.error("Vector store database is not initialized.")
                raise RuntimeError("Vector store database is not initialized.")

        def invoke(self, query):
            if not query:
                logging.error("Query string cannot be empty.")
                raise ValueError("Query string cannot be empty.")
            try:
                logging.info("Retrieving context and scores from the vector database")
                context, scores = self.vector_database.retrieve(query, top_k=self.top_k)
                if self.generate_clustered_prompts:
                    from .prompt_augmentation import generate_clustered_prompts
                    context = generate_clustered_prompts(context, embeddings=self.vector_database.embeddings)

                if not self.document_relevancy_filter:
                    logging.info("Generating answer without document relevancy filter")
                    answer = self.qa_model.answer_question(context=context, question=query)
                else:
                    logging.info("Generating answer with document relevancy filter")
                    # from .prompt_augmentation import RAGGraph
                    # graph = RAGGraph(self.qa_model)
                    # graph_out = graph.run({'question': query, 'documents': context, 'scores': scores})
                    # answer = self.qa_model.answer_question(context=graph_out['documents'],
                    #                                        question=graph_out['question'])
                    # context, scores = graph_out['documents'], graph_out['scores']
                    context = self.qa_model.grade_docs(context=context, question=query)
                    answer = self.qa_model.answer_question(context=context, query=query)

                retrieve_context = context
                new_entry = {'query': query, 'answer': answer, 'context': context}
                self.qa_history.append(new_entry)
                self.context = retrieve_context
                logging.info("Query answered successfully")
                return answer
            except Exception as e:
                logging.error(f"Error while answering query: {e}")
                raise

    class AgenticRag:
        def __init__(self, llm, vector_database=None , top_k: int = 5):
            self.llm = llm
            self.top_k = top_k
            self.vector_database = vector_database
            self.qa_history = []
            self.context = []
            # if self.vector_database is None:
            #     logging.error("Vector store database is not initialized.")
            #     raise RuntimeError("Vector store database is not initialized.")

        def run(self, query):
            grade_context = []
            if not query:
                logging.error("Query string cannot be empty.")
                raise ValueError("Query string cannot be empty.")
            else:
                if self.vector_database is not None:
                    context, scores = self.vector_database.retrieve(query, top_k=self.top_k)
                    grade_context = self.llm.grade_docs(context=context, question=query)
            # Any relevant doc? if yes -> next step(check for hallucination) , if no -> web search
            if len(grade_context) < 1:  # it means not relevant doc
                # go for web search
                logging.info("No Relevant document found, Start web search")
                # TODO add web search functionality here
                from .utils import search_duckduckgo
                print("No Relevant Context Found, Start Searching On Web...")
                results_for_searching_query = search_duckduckgo(query)
                print("Answer Base On Web Search")
                answer = self.llm.answer_question(context=results_for_searching_query, question=query)
                print("Check For Hallucination In Generated Answer Base On Web Search")

                hallucination_check_web_search_result = self.llm.check_hallucination(
                    context=results_for_searching_query, answer=answer)

                if hallucination_check_web_search_result.lower() == "yes":
                    logging.info("Hallucination detected, Regenerate the answer...")
                    # go for regenerate
                    answer = self.llm.answer_question(context=results_for_searching_query,
                                                      question=query)
                    return answer
                else:  # it means there is no hallucination
                    logging.info("Not Hallucinate")
                    return answer

            else:  # have relevant doc
                # check for hallucinating , if yes -> generate again , if no -> answer question
                answer = self.llm.answer_question(context=grade_context, question=query)
                hallucination_check = self.llm.check_hallucination(context=grade_context, answer=answer)
                if hallucination_check.lower() == "yes":
                    logging.info("Hallucination detected, Regenerate the answer...")
                    # go for regenerate
                    answer = self.llm.answer_question(context=grade_context,
                                                      question=query)
                    return answer
                else:  # it means there is no hallucination
                    logging.info("Not Hallucinate")
                    return answer

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

    # def visualize_context(self):
    #     """
    #     Visualize the context of the last query made by the user.
    #     """
    #     if not self.qa_history:
    #         print("No entries to visualize.")
    #         return
    #
    #     last_entry = self.qa_history[-1]
    #     return visualize_contexts_(last_entry['query'], last_entry['context'], last_entry['scores'])

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
