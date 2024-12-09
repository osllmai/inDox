import requests
from typing import List, Optional, cast
from loguru import logger
import sys
from indoxRag.core import Embeddings
import json

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class NerdTokenEmbedding(Embeddings):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.url = "https://api-token.nerdstudio.ai/v1/api/embeddings/"
        logger.info(f"Initialized IndoxOpenAIEmbedding with model: {model}")

    def _get_len_safe_embeddings(
        self, texts: List[str], engine: str
    ) -> List[List[float]]:
        """
        Fetch embeddings from the middle URL, ensuring text length safety.

        Args:
            texts: The list of texts to embed.
            engine: The deployment engine to use for embeddings.

        Returns:
            List of embeddings, one for each text.
        """
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        embeddings = []
        logger.info(f"Starting to fetch embeddings texts using engine: {engine}")

        for text in texts:
            data = {
                "input": text,
                "model": self.model,
            }

            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            # embedding = response.json()
            # payload = {"encoding_format": "float", "input": text, "model": engine}
            # response = requests.post(self.url, headers=headers, json=payload)
            # logger.debug(
            #     f"Response status code: {response.status_code}, Response text: {response.text}"
            # )

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data")
                if data and isinstance(data, list) and "embedding" in data[0]:
                    embedding = data[0]["embedding"]
                    embeddings.append(embedding)
                else:
                    logger.error("Embedding not found in the response.")
                    raise ValueError("Embedding not found in the response.")
            else:
                logger.error(
                    f"Failed to fetch embeddings: {response.status_code}, {response.text}"
                )
                raise Exception(
                    f"Failed to fetch embeddings: {response.status_code}, {response.text}"
                )

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        engine = cast(str, self.model)
        logger.info(f"Embedding documents")
        return self._get_len_safe_embeddings(texts, engine=engine)

    def embed_query(self, text: str) -> List[float]:
        """
        Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]
