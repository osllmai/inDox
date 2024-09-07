from typing import List, Any, Tuple
from loguru import logger
import sys
import warnings

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
        all_results = []

        for store in self.vector_stores:
            try:
                results = store.similarity_search_with_score(query, k=k)
                logger.info(f"Results from store {store}: {results}")
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error while searching in store {store}: {e}")

        # Sort results by score in descending order
        all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        # Get the top k results
        top_results = all_results[:k]

        for result in top_results:
            document, score = result
            logger.info(f"Top result: {document}, Score: {score}")

        return top_results
