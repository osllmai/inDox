import warnings
from loguru import logger
import sys
import pickle
import os

warnings.filterwarnings("ignore")

# Set up logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class KVCache:
    """
    Key-Value Cache for storing and managing preloaded knowledge.
    """

    def __init__(self, cache_dir="kv_cache"):
        """
        Initialize the cache with a directory for storing precomputed KV pairs.
        """
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def save_cache(self, key, kv_data):
        """
        Save KV cache to disk.

        Args:
            key (str): The cache key.
            kv_data (object): The data to save.
        """
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(kv_data, f)
        logger.info(f"KV cache saved: {filepath}")

    def load_cache(self, key):
        """
        Load KV cache from disk.

        Args:
            key (str): The cache key.

        Returns:
            object: The loaded KV cache data.
        """
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                kv_data = pickle.load(f)
            logger.info(f"KV cache loaded: {filepath}")
            return kv_data
        else:
            logger.warning(f"KV cache not found for key: {key}")
            return None

    def reset_cache(self):
        """
        Clear the cache directory.
        """
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            os.remove(filepath)
        logger.info("KV cache reset successfully")


class CAG:
    """
    Cache-Augmented Generation Pipeline.
    """

    def __init__(self, llm, embedding_model, cache: KVCache, max_tokens=128000):
        """
        Initialize the CAG pipeline.

        Args:
            llm (object): The LLM instance.
            embedding_model (object): The embedding model instance.
            cache (KVCache): The KV cache instance.
            max_tokens (int): The maximum tokens for the LLM context.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.cache = cache
        self.max_tokens = max_tokens
        self.loaded_kv_cache = None

    def preload_documents(self, documents, cache_key):
        """
        Precompute the KV cache from documents and save it.

        Args:
            documents (list of str): The documents to preload.
            cache_key (str): The key for the KV cache.
        """
        logger.info(f"Precomputing KV cache for {len(documents)} documents...")
        try:
            kv_cache = self.embedding_model.embed_documents(documents)
            self.cache.save_cache(cache_key, kv_cache)
            logger.info(f"Preloaded {len(documents)} documents into KV cache.")
        except Exception as e:
            logger.error(f"Error during KV cache preloading: {e}")
            raise

    def inference(self, query, cache_key):
        """
        Perform inference using the precomputed KV cache and a query.

        Args:
            query (str): The query to answer.
            cache_key (str): The key for the KV cache.

        Returns:
            str: The generated response.
        """
        if not self.loaded_kv_cache:
            logger.info(f"Loading KV cache for key: {cache_key}")
            self.loaded_kv_cache = self.cache.load_cache(cache_key)

        if not self.loaded_kv_cache:
            logger.error("KV cache is not loaded. Please preload the documents first.")
            raise RuntimeError("KV cache missing or not loaded.")

        try:
            # Generate query embedding
            logger.info("Generating query embedding...")
            query_embedding = self.embedding_model.embed_query(query)

            # Perform inference with the LLM
            logger.info("Performing inference with preloaded KV cache...")
            response = self.llm.answer_question(
                context=self.loaded_kv_cache, question=query
            )
            return response
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def reset_session(self):
        """
        Reset the session by clearing the loaded KV cache.
        """
        logger.info("Resetting session and clearing loaded KV cache...")
        self.loaded_kv_cache = None
