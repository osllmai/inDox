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


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        key: The key to look up in the dictionary.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.

    Returns:
        str: The value of the key.

    Raises:
        ValueError: If the key is not in the dictionary and no default value is
            provided or if the environment variable is not set.
    """
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )


def get_from_dict_or_env(
        data: Dict[str, Any],
        key: Union[str, List[str]],
        env_key: str,
        default: Optional[str] = None,
) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        data: The dictionary to look up the key in.
        key: The key to look up in the dictionary. This can be a list of keys to try
            in order.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.
    """
    if isinstance(key, (list, tuple)):
        for k in key:
            if k in data and data[k]:
                return data[k]

    if isinstance(key, str):
        if key in data and data[key]:
            return data[key]

    if isinstance(key, (list, tuple)):
        key_for_err = key[0]
    else:
        key_for_err = key

    return get_from_env(key_for_err, env_key, default=default)


class CohereEmbeddings(Embeddings):
    """Cohere embedding models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import CohereEmbeddings
            cohere = CohereEmbeddings(
                model="embed-english-light-v3.0",
                cohere_api_key="my-api-key"
            )
    """

    client: Any  #: :meta private:
    """Cohere client."""
    async_client: Any  #: :meta private:
    """Cohere async client."""
    model: str = "embed-english-v2.0"
    """Model name to use."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    cohere_api_key: Optional[str] = None

    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[float] = None
    """Timeout in seconds for the Cohere API request."""
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    class Config:
        extra = "forbid"

    def embed_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the embed call."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        def _embed_with_retry(**kwargs: Any) -> Any:
            return self.client.embed(**kwargs)

        return _embed_with_retry(**kwargs)

    def embed(
            self, texts: List[str], *, input_type: Optional[str] = None
    ) -> List[List[float]]:
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
        """Call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed([text], input_type="search_query")[0]
