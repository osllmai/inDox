# from typing import List, Any, Tuple
import warnings

# from .core import Document
# import logging
from .utils import show_indox_logo
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


class IndoxRetrievalAugmentation:
    def __init__(self):
        """
        Initialize the IndoxRetrievalAugmentation class
        """
        from . import __version__
        self.__version__ = __version__
        self.db = None
        self.qa_history = []
        logger.info("IndoxRetrievalAugmentation initialized")
        show_indox_logo()

    # def connect_to_vectorstore(self, vectorstore_database):
    #     """
    #     Establish a connection to the vector store database using configuration parameters.
    #     """
    #     try:
    #         self.db = vectorstore_database
    #
    #         if self.db is None:
    #             raise RuntimeError('Failed to connect to the vector store database.')
    #
    #         logger.info("Connection to the vector store database established successfully")
    #         return self.db
    #     except ValueError as ve:
    #         logger.error(f"Invalid input: {ve}")
    #         raise ValueError(f"Invalid input: {ve}")
    #     except RuntimeError as re:
    #         logger.error(f"Runtime error: {re}")
    #         raise RuntimeError(f"Runtime error: {re}")
    #     except Exception as e:
    #         logger.error(f"Failed to connect to the database due to an unexpected error: {e}")
    #         raise RuntimeError(f"Failed to connect to the database due to an unexpected error: {e}")

    # def store_in_vectorstore(self, docs: List[str]) -> Any:
    #     """
    #     Store text chunks into a vector store database.
    #     """
    #     if not docs or not isinstance(docs, list):
    #         logger.error("The `docs` parameter must be a non-empty list.")
    #         raise ValueError("The `docs` parameter must be a non-empty list.")
    #
    #     try:
    #         logger.info("Storing documents in the vector store")
    #         if self.db is not None:
    #             try:
    #                 if isinstance(docs[0], Document):
    #                     self.db.add_documents(documents=docs)
    #                 elif not isinstance(docs[0], Document):
    #                     self.db.add_texts(texts=docs)
    #                 logger.info("Document added successfully to the vector store.")
    #             except Exception as e:
    #                 logger.error(f"Failed to add document: {e}")
    #                 raise RuntimeError(f"Can't add document to the vector store: {e}")
    #         else:
    #             raise RuntimeError("The vector store database is not initialized.")
    #
    #         logger.info("Documents stored successfully")
    #         return self.db
    #     except ValueError as ve:
    #         logger.error(f"Invalid input data: {ve}")
    #         raise ValueError(f"Invalid input data: {ve}")
    #     except RuntimeError as re:
    #         logger.error(f"Runtime error while storing in the vector store: {re}")
    #         return None
    #     except Exception as e:
    #         logger.error(f"Unexpected error while storing in the vector store: {e}")
    #         return None

    class RetrievalAugmentation:
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
                logger.error("Vector store database is not initialized.")
                raise RuntimeError("Vector store database is not initialized.")

        def invoke(self, query):
            if not query:
                logger.error("Query string cannot be empty.")
                raise ValueError("Query string cannot be empty.")

            try:
                logger.info("Retrieving context and scores from the vector database")
                retrieved = self.vector_database._similarity_search_with_score(query, k=self.top_k)
                context = [d[0].page_content for d in retrieved]
                scores = [d[1] for d in retrieved]
                if self.generate_clustered_prompts:
                    from .prompt_augmentation import generate_clustered_prompts
                    context = generate_clustered_prompts(context, embeddings=self.vector_database.embeddings,
                                                         summary_model=self.qa_model)

                if not self.document_relevancy_filter:
                    logger.info("Generating answer without document relevancy filter")
                    answer = self.qa_model.answer_question(context=context, question=query)
                else:
                    logger.info("Generating answer with document relevancy filter")
                    context = self.qa_model.grade_docs(context=context, question=query)
                    answer = self.qa_model.answer_question(context=context, query=query)

                retrieve_context = context
                new_entry = {'query': query, 'answer': answer, 'context': context}
                self.qa_history.append(new_entry)
                self.context = retrieve_context
                logger.info("Query answered successfully")
                return answer
            except Exception as e:
                logger.error(f"Error while answering query: {e}")
                raise

    class AgenticRag:
        def __init__(self, llm, vector_database=None, top_k: int = 5):
            self.llm = llm
            self.top_k = top_k
            self.vector_database = vector_database
            self.qa_history = []
            self.context = []

        def run(self, query):
            grade_context = []
            if not query:
                logger.error("Query string cannot be empty.")
                raise ValueError("Query string cannot be empty.")
            else:
                if self.vector_database is not None:
                    context, scores = self.vector_database.retrieve(query, top_k=self.top_k)
                    grade_context = self.llm.grade_docs(context=context, question=query)
            # Any relevant doc? if yes -> next step(check for hallucination) , if no -> web search
            if len(grade_context) < 1:  # it means not relevant doc
                # go for web search
                logger.info("No Relevant document found, Start web search")
                from .utils import search_duckduckgo
                logger.info("No Relevant Context Found, Start Searching On Web...")
                results_for_searching_query = search_duckduckgo(query)
                logger.info("Answer Base On Web Search")
                answer = self.llm.answer_question(context=results_for_searching_query, question=query)
                logger.info("Check For Hallucination In Generated Answer Base On Web Search")

                hallucination_check_web_search_result = self.llm.check_hallucination(
                    context=results_for_searching_query, answer=answer)

                if hallucination_check_web_search_result.lower() == "yes":
                    logger.info("Hallucination detected, Regenerate the answer...")
                    # go for regenerate
                    answer = self.llm.answer_question(context=results_for_searching_query,
                                                      question=query)
                    return answer
                else:  # it means there is no hallucination
                    logger.info("Not Hallucinate")
                    return answer

            else:  # have relevant doc
                # check for hallucinating , if yes -> generate again , if no -> answer question
                answer = self.llm.answer_question(context=grade_context, question=query)
                hallucination_check = self.llm.check_hallucination(context=grade_context, answer=answer)
                if hallucination_check.lower() == "yes":
                    logger.info("Hallucination detected, Regenerate the answer...")
                    # go for regenerate
                    answer = self.llm.answer_question(context=grade_context,
                                                      question=query)
                    return answer
                else:  # it means there is no hallucination
                    logger.info("Not Hallucinate")
                    return answer
