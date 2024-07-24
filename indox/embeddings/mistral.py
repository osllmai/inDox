from typing import List, cast
from loguru import logger
import sys

from indox.core import Embeddings

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class MistralEmbedding(Embeddings):
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        from mistralai.client import MistralClient

        self.api_key = api_key
        self.model = model
        self.client = MistralClient(api_key=api_key)
        logger.info(f'Initialized MistralEmbedding with model: {model}')

    def _get_len_safe_embeddings(self, texts: List[str], engine: str) -> List[List[float]]:
        """
        Fetch embeddings from Mistral AI, ensuring text length safety.

        Args:
            texts: The list of texts to embed.
            engine: The deployment engine to use for embeddings.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        logger.info(f'Starting to fetch embeddings for texts using engine: {engine}')

        try:
            response = self.client.embeddings(
                model=engine,
                input=texts
            )
            logger.debug(f'Response: {response}')

            for embedding_object in response.data:
                embeddings.append(embedding_object.embedding)

        except Exception as e:
            logger.error(f'Failed to fetch embeddings: {e}')
            raise

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Call out to Mistral AI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        engine = cast(str, self.model)
        logger.info('Embedding documents')
        return self._get_len_safe_embeddings(texts, engine=engine)

    def embed_query(self, text: str) -> List[float]:
        """
        Call out to Mistral AI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]
