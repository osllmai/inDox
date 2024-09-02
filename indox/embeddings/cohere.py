import os
from typing import Any, Dict, List, Optional, Union, Callable
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential, before_sleep_log, retry
from indox.core import Embeddings
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")

logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")

_ALLOW_REUSE_WARNING_MESSAGE = '`allow_reuse` is deprecated and will be ignored; it should no longer be necessary'


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    import cohere

    # support v4 and v5
    retry_conditions = (
        retry_if_exception_type(cohere.error.CohereError)
        if hasattr(cohere, "error")
        else retry_if_exception_type(Exception)
    )

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_conditions,
    )





class CohereEmbeddings(Embeddings):
    """Cohere embedding models.

    To use, you should have the `cohere` python package installed, and the
    environment variable `COHERE_API_KEY` set with your API key or pass it
    as a named parameter to the constructor.
    """

    client: Any  #: :meta private:
    async_client: Any  #: :meta private:
    model: str = "embed-english-v2.0"
    truncate: Optional[str] = None
    cohere_api_key: Optional[str] = None
    max_retries: int = 3
    request_timeout: Optional[float] = None
    user_agent: str = "langchain"

    class Config:
        extra = "forbid"

    def embed_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the embed call."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        def _embed_with_retry(**kwargs: Any) -> Any:
            return self.client.embed(**kwargs)

        return _embed_with_retry(**kwargs)

    def embed(self, texts: List[str], *, input_type: Optional[str] = None) -> List[List[float]]:
        """Embed a list of texts using the Cohere model.

        Args:
            texts: The list of texts to embed.
            input_type: The input type for the embedding (e.g., search_document, search_query).

        Returns:
            List[List[float]]: A list of embeddings for each text.
        """
        embeddings = self.embed_with_retry(
            model=self.model,
            texts=texts,
            input_type=input_type,
            truncate=self.truncate,
        ).embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self.embed(texts, input_type="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed([text], input_type="search_query")[0]
