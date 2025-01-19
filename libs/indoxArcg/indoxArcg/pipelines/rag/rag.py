import sys
from typing import List, Dict, Optional, Any
from loguru import logger
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# Configure logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


@dataclass
class RetrievalResult:
    """Data class to store retrieval results"""

    content: str
    score: float = 0.0


@dataclass
class QueryResult:
    """Data class to store query results"""

    question: str
    answer: str
    context: List[str]


class BaseRetriever:
    """Base class for retrieval strategies"""

    def __init__(self, vector_store, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievalResult]:
        raise NotImplementedError


class StandardRetriever(BaseRetriever):
    """Standard vector store retrieval"""

    def retrieve(self, query: str) -> List[RetrievalResult]:
        results = self.vector_store._similarity_search_with_score(query, k=self.top_k)
        return [
            RetrievalResult(content=doc[0].page_content, score=doc[1])
            for doc in results
        ]


class MultiQueryRetriever(BaseRetriever):
    """Multi-query retrieval strategy"""

    def __init__(self, vector_store, llm, top_k: int = 5):
        super().__init__(vector_store, top_k)
        self.llm = llm

    def retrieve(self, query: str) -> List[RetrievalResult]:
        # Implementation of multi-query retrieval logic
        # This would generate multiple queries and combine results
        pass


class AnswerValidator:
    """Validates generated answers for quality and hallucination"""

    def __init__(self, llm):
        self.llm = llm

    def check_hallucination(self, answer: str, context: List[str]) -> bool:
        result = self.llm.check_hallucination(context=context, answer=answer)
        print(result)
        return result.lower() == "yes"

    def grade_relevance(self, context: List[str], query: str) -> List[str]:
        return self.llm.grade_docs(context=context, question=query)


class WebSearchFallback:
    """Handles web search when local context is insufficient"""

    def search(self, query: str) -> List[str]:
        from ...utils import search_duckduckgo

        logger.info("Performing web search for additional context")
        return search_duckduckgo(query)


class RAGError(Exception):
    """Base exception class for RAG-specific errors"""

    pass


class ContextRetrievalError(RAGError):
    """Raised when context retrieval fails"""

    pass


class AnswerGenerationError(RAGError):
    """Raised when answer generation fails"""

    pass


class RAG:
    """Main RAG pipeline orchestrator with optional smart retrieval"""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.conversation_history: Dict[int, QueryResult] = {}

    def _get_retriever(self, use_multi_query: bool, top_k: int) -> BaseRetriever:
        """Get appropriate retriever based on configuration"""
        if use_multi_query:
            return MultiQueryRetriever(self.vector_store, self.llm, top_k)
        return StandardRetriever(self.vector_store, top_k)

    def _process_context(
        self, context: List[str], query: str, use_clustering: bool
    ) -> List[str]:
        """Process and optionally cluster retrieved context"""
        if use_clustering:
            from ...prompt_augmentation import generate_clustered_prompts

            return generate_clustered_prompts(
                context,
                embeddings=self.vector_store.embeddings,
                summary_model=self.llm,
            )
        return context

    def _generate_answer(self, context: List[str], query: str) -> str:
        """Generate answer from context"""
        return self.llm.answer_question(context=context, question=query)

    def _smart_retrieve(
        self, question: str, top_k: int, min_relevance_score: float = 0.7
    ) -> List[str]:
        """Smart retrieval with validation and web fallback"""
        logger.info("Using smart retrieval")

        # Initial retrieval
        retriever = self._get_retriever(False, top_k)
        initial_results = retriever.retrieve(question)
        initial_context = [r.content for r in initial_results]

        if not initial_context:
            logger.warning("No initial context found")
            return []

        # Validate context
        validator = AnswerValidator(self.llm)
        relevance_scores = validator.grade_relevance(initial_context, question)

        # Filter by relevance
        good_context = [
            ctx
            for ctx, score in zip(initial_context, relevance_scores)
            if score >= min_relevance_score
        ]

        # Try web fallback if needed
        if len(good_context) < min(2, top_k):
            logger.info("Insufficient relevant context, trying web fallback")
            web_fallback = WebSearchFallback()
            web_results = web_fallback.search(question)

            # Validate web results
            web_scores = validator.grade_relevance(web_results, question)
            good_web_context = [
                ctx
                for ctx, score in zip(web_results, web_scores)
                if score >= min_relevance_score
            ]

            good_context.extend(good_web_context)

        return good_context[:top_k]

    def infer(
        self,
        question: str,
        top_k: int = 5,
        use_clustering: bool = False,
        use_multi_query: bool = False,
        use_smart_retrieval: bool = False,
        min_relevance_score: float = 0.7,
    ) -> str:
        """
        Main query method with configurable retrieval strategy

        Args:
            question: The query string
            top_k: Number of documents to retrieve
            use_clustering: Whether to use clustering for context processing
            use_multi_query: Whether to use multi-query retrieval
            use_smart_retrieval: Whether to use smart retrieval with validation and web fallback
            min_relevance_score: Minimum relevance score for smart retrieval (0-1)
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            # Get context based on retrieval strategy
            if use_smart_retrieval:
                context = self._smart_retrieve(question, top_k, min_relevance_score)
            else:
                retriever = self._get_retriever(use_multi_query, top_k)
                results = retriever.retrieve(question)
                context = [r.content for r in results]

            if not context:
                raise ContextRetrievalError(
                    "No relevant context found for the question"
                )

            # Process context
            context = self._process_context(context, question, use_clustering)

            # Generate answer
            answer = self._generate_answer(context, question)

            # Store in conversation history
            self.conversation_history[len(self.conversation_history)] = QueryResult(
                question=question, answer=answer, context=context
            )

            return answer

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise
