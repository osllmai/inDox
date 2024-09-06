from typing import List, Dict, Any, Optional,Tuple
from indox.core import Document
from loguru import logger
import sys
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class MultiVectorRetriever:
    def __init__(self, vector_store_classes: List, embedding_function: Any, llm: Any,
                 collection_name: Optional[str] = None, keyspace: Optional[List[str]] = None):
        self.vector_stores: Dict[str, Any] = {}
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.keyspace = keyspace

        for store_class in vector_store_classes:
            try:
                init_params = store_class.__init__.__code__.co_varnames

                init_args = {}
                if 'collection_name' in init_params:
                    init_args['collection_name'] = self.collection_name
                if 'embedding_function' in init_params:
                    init_args['embedding_function'] = self.embedding_function
                if 'keyspace' in init_params:
                    init_args['keyspace'] = self.keyspace

                store_instance = store_class(**init_args)
                self.vector_stores[store_class.__name__] = store_instance
            except Exception as e:
                logger.error(f"Failed to initialize {store_class.__name__}: {e}")


    def add(self, docs):
        for store_name, store_instance in self.vector_stores.items():
            try:
                store_instance.add(docs=docs)
            except Exception as e:
                logger.error(f"Failed to add documents to {store_name}: {e}")

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        results = []

        for store_name, store_instance in self.vector_stores.items():
            try:
                search_results = store_instance.similarity_search_with_score(query, k=k)
                logger.info(f"Search results from {store_name}: {search_results}")
                results.extend(search_results)
            except Exception as e:
                logger.error(f"Failed to search in {store_name}: {e}")

        return results


