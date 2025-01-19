from typing import List, Any, Tuple
from loguru import logger
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

# Set up logging
warnings.filterwarnings("ignore")
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class MultiVectorRetriever:
    def __init__(self, vector_stores: List[Any]):
        self.vector_stores = vector_stores

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Run similarity search across multiple vector stores and return results with scores.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 5.

        Returns:
            List[Tuple[Any, float]]: List of objects (e.g., documents) most similar to
            the query text and the corresponding similarity score as a float for each.
            Higher score represents more similarity.

        Raises:
            Exception: Logs any exception raised during the similarity search process.

        Notes:
            The method executes similarity searches in parallel across all vector stores
            using a thread pool executor and combines results from each store. Results
            are sorted by similarity score in descending order.
        """
        all_results = []

        def search_in_store(store):
            try:
                results = store.similarity_search_with_score(query, k=k)
                return results
            except Exception as e:
                logger.error(f"Error while searching in store {store}: {e}")
                return []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(search_in_store, store) for store in self.vector_stores]
            for future in futures:
                all_results.extend(future.result())

        combined_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        # logger.info(f"Combined sorted results from all vector stores: {combined_results}")
        return combined_results[:k]
