import warnings
from loguru import logger
import sys
from typing import List

warnings.filterwarnings("ignore")

logger.remove()
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")

class MultiQueryRetrieval:
    """
    A class that implements multi-query retrieval for enhanced information gathering.

    This class generates multiple queries from an original query, retrieves relevant
    information for each generated query, and combines the results to produce a final response.

    """

    def __init__(self, llm, vector_database, top_k: int = 3):
        """
        Initialize the MultiQueryRetrieval instance.

        Args:
            llm: The language model to use for query generation and response synthesis.
            vector_database: The vector database to use for information retrieval.
            top_k (int): The number of top results to retrieve for each query. Defaults to 3.
        """
        self.llm = llm
        self.vector_database = vector_database
        self.top_k = top_k

    def generate_queries(self, original_query: str) -> List[str]:
        """
        Generate multiple queries from the original query.

        Args:
            original_query (str): The original user query.

        Returns:
            List[str]: A list of generated queries.
        """
        prompt = f"Generate 3 different queries to gather information for answering the following question: {original_query}"
        response = self.llm.chat(prompt=prompt)
        return [q.strip() for q in response.split('\n') if q.strip()]

    def retrieve_information(self, queries: List[str]) -> List[str]:
        """
        Retrieve relevant information for each generated query.

        Args:
            queries (List[str]): A list of queries to use for information retrieval.

        Returns:
            List[str]: A list of relevant passages retrieved from the vector database.
        """
        all_relevants = []
        for query in queries:
            retrieved = self.vector_database._similarity_search_with_score(query, k=self.top_k)
            relevants = [d[0].page_content for d in retrieved]
            all_relevants.extend(relevants)
        return all_relevants

    def generate_response(self, original_query: str, context: List[str]) -> str:
        """
        Generate a final response based on the original query and retrieved context.

        Args:
            original_query (str): The original user query.
            context (List[str]): A list of relevant passages to use as context.

        Returns:
            str: The generated response.
        """
        combined_context = "\n".join(context)
        prompt = f"Based on the following information, answer the question: {original_query}\n\nContext: {combined_context}"
        return self.llm.chat(prompt=prompt)

    def run(self, query: str) -> str:
        """
        Execute the full multi-query retrieval process.

        This method orchestrates the entire process of query generation, information retrieval,
        and response generation.

        Args:
            query (str): The original user query.

        Returns:
            str: The final generated response.
        """
        logger.info(f"Running multi-query retrieval for: {query}")
        generated_queries = self.generate_queries(query)
        logger.info(f"Generated queries: {generated_queries}")

        relevants = self.retrieve_information(generated_queries)
        logger.info(f"Retrieved {len(relevants)} relevant passages")

        response = self.generate_response(query, relevants)
        logger.info("Generated final response")

        return response