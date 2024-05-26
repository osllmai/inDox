import requests
from typing import List, Optional, cast
import requests
import numpy as np
from typing import List


def OpenAiEmbedding(model, openai_api_key):
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=openai_api_key)
    return embeddings


def HuggingFaceEmbedding(model):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=model)
    return embeddings


# multi-qa-mpnet-base-cos-v1

class IndoxOpenAIEmbedding:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.middle_url = "http://5.78.55.161/api/embedding/generate/"

    def _get_len_safe_embeddings(self, texts: List[str], engine: str) -> List[List[float]]:
        """
        Fetch embeddings from the middle URL, ensuring text length safety.

        Args:
            texts: The list of texts to embed.
            engine: The deployment engine to use for embeddings.

        Returns:
            List of embeddings, one for each text.
        """
        headers = {
            'accept': '*/*',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        embeddings = []

        for text in texts:
            payload = {
                "encoding_format": "float",
                "input": text,
                "model": engine
            }

            response = requests.post(self.middle_url, headers=headers, json=payload)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get('data')
                if data and isinstance(data, list) and 'embedding' in data[0]:
                    embedding = data[0]['embedding']
                    embeddings.append(embedding)  # Directly append the embedding list
                else:
                    raise ValueError("Embedding not found in the response.")
            else:
                raise Exception(f"Failed to fetch embeddings: {response.status_code}, {response.text}")

        return embeddings

    def embed_documents(
            self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """
        Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                        specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.model)
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
