import requests
from typing import List, Optional, cast
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


def OpenAiEmbedding(model, api_key):
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=model, api_key=api_key)
    logging.info(f'Initialized OpenAI embeddings with model: {model}')
    return embeddings


def HuggingFaceEmbedding(model):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=model)
    logging.info(f'Initialized HuggingFace embeddings with model: {model}')
    return embeddings


def MistralEmbedding(api_key):
    from langchain_mistralai import MistralAIEmbeddings
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key
    )
    logging.info(f'Initialized Mistral embeddings')
    return embeddings


class IndoxApiEmbedding:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.middle_url = "http://5.78.55.161/api/embedding/generate/"
        logging.info(f'Initialized IndoxOpenAIEmbedding with model: {model} and middle URL: {self.middle_url}')

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
        logging.info(f'Starting to fetch embeddings for {len(texts)} texts using engine: {engine}')

        for text in texts:
            payload = {
                "encoding_format": "float",
                "input": text,
                "model": engine
            }
            logging.debug(f'Request payload: {payload}')

            response = requests.post(self.middle_url, headers=headers, json=payload)
            logging.debug(f'Response status code: {response.status_code}, Response text: {response.text}')

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get('data')
                if data and isinstance(data, list) and 'embedding' in data[0]:
                    embedding = data[0]['embedding']
                    embeddings.append(embedding)
                else:
                    logging.error('Embedding not found in the response.')
                    raise ValueError("Embedding not found in the response.")
            else:
                logging.error(f'Failed to fetch embeddings: {response.status_code}, {response.text}')
                raise Exception(f"Failed to fetch embeddings: {response.status_code}, {response.text}")

        return embeddings

    def embed_documents(self, texts: List[str], chunk_size: Optional[int] = 0) -> List[List[float]]:
        """
        Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                        specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        engine = cast(str, self.model)
        logging.info(f'Embedding documents with chunk size: {chunk_size}')
        return self._get_len_safe_embeddings(texts, engine=engine)

    def embed_query(self, text: str) -> List[float]:
        """
        Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        logging.info(f'Embedding query text: {text}')
        return self.embed_documents([text])[0]
