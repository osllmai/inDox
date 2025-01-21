# # import warnings
# # from loguru import logger
# # import sys
# # import pickle
# # import os

# # warnings.filterwarnings("ignore")

# # # Set up logging
# # logger.remove()
# # logger.add(
# #     sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
# # )
# # logger.add(
# #     sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
# # )


# # class KVCache:
# #     """
# #     Key-Value Cache for storing and managing preloaded knowledge.
# #     """

# #     def __init__(self, cache_dir="kv_cache"):
# #         """
# #         Initialize the cache with a directory for storing precomputed KV pairs.
# #         """
# #         self.cache_dir = cache_dir
# #         if not os.path.exists(self.cache_dir):
# #             os.makedirs(self.cache_dir)

# #     def save_cache(self, key, kv_data):
# #         """
# #         Save KV cache to disk.

# #         Args:
# #             key (str): The cache key.
# #             kv_data (object): The data to save.
# #         """
# #         filepath = os.path.join(self.cache_dir, f"{key}.pkl")
# #         with open(filepath, "wb") as f:
# #             pickle.dump(kv_data, f)
# #         logger.info(f"KV cache saved: {filepath}")

# #     def load_cache(self, key):
# #         """
# #         Load KV cache from disk.

# #         Args:
# #             key (str): The cache key.

# #         Returns:
# #             object: The loaded KV cache data.
# #         """
# #         filepath = os.path.join(self.cache_dir, f"{key}.pkl")
# #         if os.path.exists(filepath):
# #             with open(filepath, "rb") as f:
# #                 kv_data = pickle.load(f)
# #             logger.info(f"KV cache loaded: {filepath}")
# #             return kv_data
# #         else:
# #             logger.warning(f"KV cache not found for key: {key}")
# #             return None

# #     def reset_cache(self):
# #         """
# #         Clear the cache directory.
# #         """
# #         for filename in os.listdir(self.cache_dir):
# #             filepath = os.path.join(self.cache_dir, filename)
# #             os.remove(filepath)
# #         logger.info("KV cache reset successfully")


# # class CAG:
# #     """
# #     Cache-Augmented Generation Pipeline.
# #     """

# #     def __init__(self, llm, embedding_model, cache: KVCache, max_tokens=128000):
# #         """
# #         Initialize the CAG pipeline.

# #         Args:
# #             llm (object): The LLM instance.
# #             embedding_model (object): The embedding model instance.
# #             cache (KVCache): The KV cache instance.
# #             max_tokens (int): The maximum tokens for the LLM context.
# #         """
# #         self.llm = llm
# #         self.embedding_model = embedding_model
# #         self.cache = cache
# #         self.max_tokens = max_tokens
# #         self.loaded_kv_cache = None

# #     def preload_documents(self, documents, cache_key):
# #         """
# #         Precompute the KV cache from documents and save it.

# #         Args:
# #             documents (list of str): The documents to preload.
# #             cache_key (str): The key for the KV cache.
# #         """
# #         logger.info(f"Precomputing KV cache for {len(documents)} documents...")
# #         try:
# #             kv_cache = self.embedding_model.embed_documents(documents)
# #             self.cache.save_cache(cache_key, kv_cache)
# #             logger.info(f"Preloaded {len(documents)} documents into KV cache.")
# #         except Exception as e:
# #             logger.error(f"Error during KV cache preloading: {e}")
# #             raise

# #     def inference(self, query, cache_key):
# #         """
# #         Perform inference using the precomputed KV cache and a query.

# #         Args:
# #             query (str): The query to answer.
# #             cache_key (str): The key for the KV cache.

# #         Returns:
# #             str: The generated response.
# #         """
# #         if not self.loaded_kv_cache:
# #             logger.info(f"Loading KV cache for key: {cache_key}")
# #             self.loaded_kv_cache = self.cache.load_cache(cache_key)

# #         if not self.loaded_kv_cache:
# #             logger.error("KV cache is not loaded. Please preload the documents first.")
# #             raise RuntimeError("KV cache missing or not loaded.")

# #         try:
# #             # Generate query embedding
# #             logger.info("Generating query embedding...")
# #             query_embedding = self.embedding_model.embed_query(query)

# #             # Perform inference with the LLM
# #             logger.info("Performing inference with preloaded KV cache...")
# #             response = self.llm.answer_question(
# #                 context=self.loaded_kv_cache, question=query
# #             )
# #             return response
# #         except Exception as e:
# #             logger.error(f"Error during inference: {e}")
# #             raise

# #     def reset_session(self):
# #         """
# #         Reset the session by clearing the loaded KV cache.
# #         """
# #         logger.info("Resetting session and clearing loaded KV cache...")
# #         self.loaded_kv_cache = None

# import warnings
# from loguru import logger
# import sys
# import pickle
# import os
# import numpy as np
# from typing import List, Dict, Any

# warnings.filterwarnings("ignore")

# # Set up logging
# logger.remove()
# logger.add(
#     sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
# )
# logger.add(
#     sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
# )


# class KVCache:
#     """
#     Key-Value Cache for storing and managing preloaded knowledge.
#     """

#     def __init__(self, cache_dir="kv_cache"):
#         self.cache_dir = cache_dir
#         if not os.path.exists(self.cache_dir):
#             os.makedirs(self.cache_dir)

#     def save_cache(self, key, kv_data):
#         filepath = os.path.join(self.cache_dir, f"{key}.pkl")
#         with open(filepath, "wb") as f:
#             pickle.dump(kv_data, f)
#         logger.info(f"KV cache saved: {filepath}")

#     def load_cache(self, key):
#         filepath = os.path.join(self.cache_dir, f"{key}.pkl")
#         if os.path.exists(filepath):
#             with open(filepath, "rb") as f:
#                 kv_data = pickle.load(f)
#             logger.info(f"KV cache loaded: {filepath}")
#             return kv_data
#         else:
#             logger.warning(f"KV cache not found for key: {key}")
#             return None

#     def reset_cache(self):
#         for filename in os.listdir(self.cache_dir):
#             filepath = os.path.join(self.cache_dir, filename)
#             os.remove(filepath)
#         logger.info("KV cache reset successfully")


# class CacheEntry:
#     """
#     Structure to hold both text content and its embedding.
#     """

#     def __init__(self, text: str, embedding: np.ndarray):
#         self.text = text
#         self.embedding = embedding


# class CAG:
#     """
#     Cache-Augmented Generation Pipeline with semantic search.
#     """

#     def __init__(
#         self,
#         llm,
#         embedding_model,
#         cache: KVCache,
#         # top_k: int = 5,
#         # similarity_threshold: float = 0.5,
#     ):
#         """
#         Initialize the CAG pipeline.

#         Args:
#             llm: The LLM instance
#             embedding_model: The embedding model instance
#             cache (KVCache): The KV cache instance
#             top_k (int): Number of most similar chunks to retrieve
#             similarity_threshold (float): Minimum similarity score (0-1) for inclusion
#         """
#         self.llm = llm
#         self.embedding_model = embedding_model
#         self.cache = cache
#         self.loaded_kv_cache = None
#         # self.top_k = top_k
#         # self.similarity_threshold = similarity_threshold

#     def compute_similarity(
#         self, query_embedding: np.ndarray, doc_embedding: np.ndarray
#     ) -> float:
#         """
#         Compute cosine similarity between query and document embeddings.
#         """
#         return np.dot(query_embedding, doc_embedding) / (
#             np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
#         )

#     def get_relevant_context(
#         self, query_embedding: np.ndarray, cache_entries: List[CacheEntry]
#     ) -> List[str]:
#         """
#         Retrieve most relevant context chunks based on semantic similarity.
#         """
#         similarities = [
#             (entry, self.compute_similarity(query_embedding, entry.embedding))
#             for entry in cache_entries
#         ]

#         # Sort by similarity score
#         similarities.sort(key=lambda x: x[1], reverse=True)

#         # Filter by threshold and take top k
#         relevant_chunks = [
#             entry.text
#             for entry, score in similarities[: self.top_k]
#             if score >= self.similarity_threshold
#         ]

#         logger.info(f"Selected {len(relevant_chunks)} relevant chunks from cache")

#         # Log similarity scores for debugging
#         for i, (entry, score) in enumerate(similarities[: self.top_k]):
#             logger.debug(f"Chunk {i+1} similarity score: {score:.3f}")

#         return relevant_chunks

#     def preload_documents(self, documents: List[str], cache_key: str):
#         """
#         Precompute the KV cache from pre-chunked documents and save it.

#         Args:
#             documents (List[str]): Pre-chunked documents
#             cache_key (str): The key for the KV cache
#         """
#         logger.info(f"Precomputing KV cache for {len(documents)} document chunks...")
#         try:
#             # Create cache entries with both text and embeddings
#             cache_entries = []
#             for chunk in documents:
#                 embedding = self.embedding_model.embed_query(chunk)
#                 cache_entries.append(CacheEntry(chunk, embedding))

#             self.cache.save_cache(cache_key, cache_entries)
#             logger.info(f"Preloaded {len(cache_entries)} document chunks into KV cache")
#         except Exception as e:
#             logger.error(f"Error during KV cache preloading: {e}")
#             raise

#     def infer(
#         self,
#         query: str,
#         cache_key: str,
#         top_k: int = 5,
#         similarity_threshold: float = 0.5,
#     ) -> str:
#         """
#         Perform inference using the precomputed KV cache and a query.
#         """
#         self.top_k = top_k
#         self.similarity_threshold = similarity_threshold
#         if not self.loaded_kv_cache:
#             logger.info(f"Loading KV cache for key: {cache_key}")
#             self.loaded_kv_cache = self.cache.load_cache(cache_key)

#         if not self.loaded_kv_cache:
#             logger.error("KV cache is not loaded. Please preload the documents first.")
#             raise RuntimeError("KV cache missing or not loaded.")

#         try:
#             # Generate query embedding
#             logger.info("Generating query embedding...")
#             query_embedding = self.embedding_model.embed_query(query)

#             # Get relevant context using semantic search
#             logger.info("Retrieving relevant context...")
#             relevant_context = self.get_relevant_context(
#                 query_embedding, self.loaded_kv_cache
#             )

#             # Perform inference with filtered context
#             logger.info("Performing inference with filtered context...")
#             response = self.llm.answer_question(
#                 context=relevant_context, question=query
#             )
#             return response
#         except Exception as e:
#             logger.error(f"Error during inference: {e}")
#             raise

#     def reset_session(self):
#         """
#         Reset the session by clearing the loaded KV cache.
#         """
#         logger.info("Resetting session and clearing loaded KV cache...")
#         self.loaded_kv_cache = None

import warnings
from loguru import logger
import sys
import pickle
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def save_cache(self, key, kv_data):
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(kv_data, f)
        logger.info(f"KV cache saved: {filepath}")

    def load_cache(self, key):
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
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            os.remove(filepath)
        logger.info("KV cache reset successfully")


class CacheEntry:
    """
    Structure to hold text content and optionally its embedding.
    """

    def __init__(self, text: str, embedding: Optional[np.ndarray] = None):
        self.text = text
        self.embedding = embedding


class CAG:
    """
    Cache-Augmented Generation Pipeline with optional embedding-based similarity search.
    """

    def __init__(
        self,
        llm,
        embedding_model: Optional[Any] = None,  # Make embedding_model optional
        cache: Optional[KVCache] = None,
    ):
        """
        Initialize the CAG pipeline.

        Args:
            llm: The LLM instance
            embedding_model: The embedding model instance (optional)
            cache (KVCache): The KV cache instance (optional)
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.cache = cache if cache else KVCache()  # Default cache if not provided
        self.use_embedding = embedding_model is not None  # Auto-set use_embedding
        self.loaded_kv_cache = None

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between query and document embeddings.
        """
        return np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )

    def text_based_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute similarity between query and documents using TF-IDF.
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query] + documents)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return similarities

    def get_relevant_context(
        self,
        query: str,
        cache_entries: List[CacheEntry],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[str]:
        """
        Retrieve most relevant context chunks based on similarity.
        """
        if self.use_embedding:
            # Embedding-based similarity search
            query_embedding = self.embedding_model.embed_query(query)
            similarities = [
                (entry, self.compute_similarity(query_embedding, entry.embedding))
                for entry in cache_entries
                if entry.embedding is not None
            ]
        else:
            # Text-based similarity search
            document_texts = [entry.text for entry in cache_entries]
            similarities = [
                (entry, score)
                for entry, score in zip(
                    cache_entries, self.text_based_similarity(query, document_texts)
                )
            ]

        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and take top k
        relevant_chunks = [
            entry.text
            for entry, score in similarities[:top_k]
            if score >= similarity_threshold
        ]

        logger.info(f"Selected {len(relevant_chunks)} relevant chunks from cache")
        return relevant_chunks

    def preload_documents(self, documents: List[str], cache_key: str):
        """
        Precompute the KV cache from pre-chunked documents and save it.
        """
        logger.info(f"Precomputing KV cache for {len(documents)} document chunks...")
        try:
            # Create cache entries with text and optionally embeddings
            cache_entries = []
            for chunk in documents:
                if self.use_embedding:
                    embedding = self.embedding_model.embed_query(chunk)
                    cache_entries.append(CacheEntry(chunk, embedding))
                else:
                    cache_entries.append(CacheEntry(chunk))

            self.cache.save_cache(cache_key, cache_entries)
            logger.info(f"Preloaded {len(cache_entries)} document chunks into KV cache")
        except Exception as e:
            logger.error(f"Error during KV cache preloading: {e}")
            raise

    def infer(
        self,
        query: str,
        cache_key: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> str:
        """
        Perform inference using the precomputed KV cache and a query.
        """
        if not self.loaded_kv_cache:
            logger.info(f"Loading KV cache for key: {cache_key}")
            self.loaded_kv_cache = self.cache.load_cache(cache_key)

        if not self.loaded_kv_cache:
            logger.error("KV cache is not loaded. Please preload the documents first.")
            raise RuntimeError("KV cache missing or not loaded.")

        try:
            # Retrieve relevant context
            logger.info("Retrieving relevant context...")
            relevant_context = self.get_relevant_context(
                query, self.loaded_kv_cache, top_k, similarity_threshold
            )

            # Perform inference with filtered context
            logger.info("Performing inference with filtered context...")
            response = self.llm.answer_question(
                context=relevant_context, question=query
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
