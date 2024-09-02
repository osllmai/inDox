# from .embedding_models import HuggingFaceEmbedding, IndoxApiEmbedding, MistralEmbedding
from .openai import OpenAiEmbedding
from .mistral import MistralEmbedding
from .huggingface import HuggingFaceEmbedding
from .indox_api import IndoxApiEmbedding
from .ollama import OllamaEmbeddings
from .clarifai import ClarifaiEmbeddings
from .gpt4all import GPT4AllEmbeddings
from .cohere import CohereEmbeddings
from .elastic_search import ElasticsearchEmbeddings
from .azure_openai import AzureOpenAIEmbeddings