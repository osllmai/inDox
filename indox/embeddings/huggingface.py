from typing import List
from loguru import logger
import sys

from indox.core import Embeddings

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
logger.add(sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")


class HuggingFaceEmbedding(Embeddings):
    def __init__(self, model: str, api_key: str):
        from sentence_transformers import SentenceTransformer

        self.model = model
        self.api_key = api_key
        self.model = SentenceTransformer(model, use_auth_token=api_key)
        logger.info(f'Initialized HuggingFaceEmbedding with model: {model}')

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetch embeddings from Hugging Face models, ensuring text length safety.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        logger.info(f'Starting to fetch embeddings for texts using model: {self.model}')
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True).tolist()
        except Exception as e:
            logger.error(f'Failed to fetch embeddings: {e}')
            raise

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Call out to Hugging Face model for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        logger.info('Embedding documents')
        return self._get_len_safe_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Call out to Hugging Face model for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]
