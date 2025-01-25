import warnings
from loguru import logger
import sys
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from .kvcache import KVCache
from .conversation_session import ConversationSession


warnings.filterwarnings("ignore")

# Set up logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class AnswerValidator:
    """Validates generated answers for quality and hallucination"""

    def __init__(self, llm):
        self.llm = llm

    def check_hallucination(self, answer: str, context: List[str]) -> bool:
        result = self.llm.check_hallucination(context=context, answer=answer)
        return result

    def grade_relevance(self, context: List[str], query: str) -> List[str]:
        return self.llm.grade_docs(context=context, question=query)


class WebSearchFallback:
    """Handles web search when local context is insufficient"""

    def search(self, query: str) -> List[str]:
        from ...utils import search_duckduckgo

        logger.info("Performing web search for additional context")
        return search_duckduckgo(query)


class CacheEntry:
    """
    Structure to hold text content and optionally its embedding.
    """

    def __init__(self, text: str, embedding: Optional[np.ndarray] = None):
        self.text = text
        self.embedding = embedding


class CAG:
    """
    Cache-Augmented Generation Pipeline with multi-query and smart retrieval.
    """

    def __init__(
        self,
        llm,
        embedding_model: Optional[Any] = None,
        cache: Optional[KVCache] = None,
        default_similarity_search_type: str = "tfidf",  # Default similarity search type
    ):
        """
        Initialize the CAG pipeline.

        Args:
            llm: The LLM instance
            embedding_model: The embedding model instance (optional)
            cache (KVCache): The KV cache instance (optional)
            default_similarity_search_type (str): Default similarity search type. Options: "tfidf", "bm25", "jaccard"
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.cache = cache if cache else KVCache()  # Default cache if not provided
        self.use_embedding = embedding_model is not None  # Auto-set use_embedding
        self.loaded_kv_cache = None
        self.default_similarity_search_type = default_similarity_search_type.lower()
        self.session = ConversationSession()
        # Validate default_similarity_search_type
        if self.default_similarity_search_type not in ["tfidf", "bm25", "jaccard"]:
            raise ValueError(
                "Invalid default_similarity_search_type. Choose from 'tfidf', 'bm25', or 'jaccard'."
            )

    def _text_based_similarity(
        self, query: str, documents: List[str], similarity_search_type: str
    ) -> List[float]:
        """
        Compute similarity between query and documents based on the selected similarity search type.
        """
        if similarity_search_type == "tfidf":
            return self._tfidf_similarity(query, documents)
        elif similarity_search_type == "bm25":
            return self._bm25_similarity(query, documents)
        elif similarity_search_type == "jaccard":
            return self._jaccard_similarity(query, documents)
        else:
            raise ValueError(
                f"Unsupported similarity search type: {similarity_search_type}"
            )

    def _tfidf_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute similarity between query and documents using TF-IDF.
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query] + documents)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return similarities

    def _bm25_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute similarity between query and documents using BM25.
        """
        from rank_bm25 import BM25Okapi

        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        return scores

    def _jaccard_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute similarity between query and documents using Jaccard similarity.
        """
        query_tokens = set(query.split())
        similarities = []
        for doc in documents:
            doc_tokens = set(doc.split())
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            jaccard_score = len(intersection) / len(union) if union else 0
            similarities.append(jaccard_score)
        return similarities

    def _get_relevant_context(
        self,
        query: str,
        cache_entries: List[CacheEntry],
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        similarity_search_type: str = "tfidf",  # Default to TF-IDF
    ) -> List[str]:
        """
        Retrieve most relevant context chunks based on similarity.
        """
        if self.use_embedding:
            # Embedding-based similarity search
            query_embedding = self.embedding_model.embed_query(query)
            similarities = [
                (entry, self._compute_similarity(query_embedding, entry.embedding))
                for entry in cache_entries
                if entry.embedding is not None
            ]
        else:
            # Text-based similarity search
            document_texts = [entry.text for entry in cache_entries]
            similarities = [
                (entry, score)
                for entry, score in zip(
                    cache_entries,
                    self._text_based_similarity(
                        query, document_texts, similarity_search_type
                    ),
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

    def _compute_similarity(
        self, query_embedding: np.ndarray, doc_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between query and document embeddings.
        """
        return np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )

    def multi_query_retrieval(self, query: str, top_k: int = 5) -> List[str]:
        """
        Generate multiple queries and retrieve context for each.
        """
        from ...tools import MultiQueryRetrieval  # Import your multi-query tool

        logger.info("Performing multi-query retrieval")
        multi_query = MultiQueryRetrieval(self.llm, self.cache, top_k)
        return multi_query.run(query)

    def smart_retrieve(
        self,
        query: str,
        cache_entries: List[CacheEntry],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        similarity_search_type: Optional[str] = None,  # New parameter
    ) -> List[str]:
        """
        Smart retrieval with validation and fallback mechanisms.
        """
        logger.info("Using smart retrieval")

        # Initial retrieval
        relevant_context = self._get_relevant_context(
            query, cache_entries, top_k, similarity_threshold, similarity_search_type
        )

        if not relevant_context:
            logger.warning("No relevant context found in cache")
            return self._handle_web_fallback(query, top_k)

        # Grade documents using LLM's relevance check
        try:
            validator = AnswerValidator(self.llm)
            graded_context = validator.grade_relevance(relevant_context, query)
            if not graded_context:
                logger.info("No relevant documents found after grading")
                return self._handle_web_fallback(query, top_k)
        except Exception as e:
            logger.error(f"Error in document grading: {str(e)}")
            graded_context = relevant_context

        # Ensure we don't exceed top_k
        graded_context = graded_context[:top_k]

        # Check for hallucination if we have context
        if graded_context:
            try:
                answer = self.llm.answer_question(
                    context=graded_context, question=query
                )
                hallucination_result = validator.check_hallucination(
                    context=graded_context, answer=answer
                )
                if hallucination_result.lower() == "yes":
                    logger.info("Hallucination detected, regenerating answer")
                    answer = self.llm.answer_question(
                        context=graded_context, question=query
                    )
            except Exception as e:
                logger.error(
                    f"Error in answer generation or hallucination check: {str(e)}"
                )

        return graded_context

    def _handle_web_fallback(self, query: str, top_k: int) -> List[str]:
        """
        Handle web search fallback when cache retrieval is insufficient.
        """
        try:
            web_fallback = WebSearchFallback()
            web_results = web_fallback.search(query)

            if not web_results:
                logger.warning("No results from web fallback")
                return []

            # Grade web results using LLM's relevance check
            validator = AnswerValidator(self.llm)
            try:
                graded_web_context = validator.grade_relevance(web_results, query)
            except Exception as e:
                logger.error(f"Error grading web results: {str(e)}")
                graded_web_context = web_results

            return graded_web_context[:top_k]

        except Exception as e:
            logger.error(f"Web fallback failed: {str(e)}")
            return []

    def infer(
        self,
        query: str,
        cache_key: str,
        context_strategy: str = "recent",
        context_turns: int = 3,
        top_k: int = 5,
        similarity_threshold: float = 0.2,
        web_search: bool = False,
        similarity_search_type: Optional[str] = None,
        smart_retrieval: bool = False,
        use_multi_query: bool = False,
    ) -> str:
        """
        Perform cache-augmented inference with multiple retrieval strategies

        Args:
            query: User's input question/request
            cache_key: Identifier for preloaded knowledge cache
            context_strategy: History handling ("recent", "relevant", "full")
            context_turns: Recent turns to consider
            top_k: Number of knowledge chunks to retrieve
            similarity_threshold: Minimum similarity score
            web_search: Enable web search fallback
            similarity_search_type: Override similarity algorithm
            smart_retrieval: Use validation-enhanced retrieval
            use_multi_query: Use multi-query expansion

        Returns:
            str: Generated response
        """
        # Validate parameters
        if not query.strip():
            raise ValueError("Query cannot be empty")

        valid_strategies = ["recent", "relevant", "full"]
        if context_strategy not in valid_strategies:
            raise ValueError(f"Invalid context_strategy: {context_strategy}")

        try:
            # 1. Get conversation context
            if context_strategy == "recent":
                session_context = self.session.get_recent_context(context_turns)
            elif context_strategy == "relevant":
                session_context = self.session.get_relevant_context(
                    query, fixed_last_n=context_turns, top_k=top_k
                )
            else:  # full
                session_context = self.session.get_full_conversation()

            # 2. Load knowledge cache
            if not self.loaded_kv_cache:
                logger.info(f"Loading knowledge cache: {cache_key}")
                self.loaded_kv_cache = self.cache.load_cache(cache_key)
                if not self.loaded_kv_cache:
                    raise RuntimeError(f"Cache {cache_key} not found")

            # 3. Determine retrieval method
            similarity_search_type = (
                similarity_search_type or self.default_similarity_search_type
            )
            retrieval_query = (
                f"{session_context}\n{query}" if session_context else query
            )

            if smart_retrieval:
                logger.info("Using smart retrieval with validation")
                cached_context = self.smart_retrieve(
                    query=retrieval_query,
                    cache_entries=self.loaded_kv_cache,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    similarity_search_type=similarity_search_type,
                )
            elif use_multi_query:
                logger.info("Using multi-query retrieval")
                cached_context = self.multi_query_retrieval(
                    query=retrieval_query, top_k=top_k
                )
            else:
                cached_context = self._get_relevant_context(
                    query=retrieval_query,
                    cache_entries=self.loaded_kv_cache,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    similarity_search_type=similarity_search_type,
                )

            # 4. Web search fallback
            web_results = []
            if web_search:
                logger.info("Initiating web search fallback")
                web_fallback = WebSearchFallback()
                web_results = web_fallback.search(query)

                if not web_results:
                    logger.warning("No results from web fallback")
                    return []

            # 5. Combine contexts
            final_context = [
                "Conversation History:",
                session_context,
                "\nRetrieved Knowledge:",
                *cached_context,
                *web_results,
            ]
            final_context_str = "\n".join(filter(None, final_context))

            logger.debug(f"Final context:\n{final_context_str}")

            # 6. Generate and validate response
            response = self.llm.answer_question(
                context=final_context_str, question=query
            )

            # 7. Update conversation history
            self.session.add_to_history(query, response)

            return response

        except json.JSONDecodeError as e:
            logger.error(f"JSON serialization error: {str(e)}")
            return "Error: Failed to process context"
        except RuntimeError as e:
            logger.error(f"Cache error: {str(e)}")
            return "Error: Knowledge base unavailable"
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return "Error: Failed to generate response"
