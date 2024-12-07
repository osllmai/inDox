# Embedding Models

Indox currently supports various embedding models. Below is a list of supported models, along with instructions on how to use each one:

1. **OpenAI Embedding Model**
2. **Hugging Face Embedding Models**
3. **IndoxApi Embedding Models**
4. **Mistral Embedding Models**
5. **Clarifai Embedding Models**
6. **Cohere Embedding Models**
7. **Elasticsearch Embedding Models**
8. **GPT4All Embedding Models**
9. **Ollama Embedding Models**

## Using OpenAI Embedding Model

To use the OpenAI embedding model, follow these steps:

1. Install the OpenAI Python package:

   ```bash
   pip install openai
   ```

2. Import necessary libraries and load environment variables:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
   ```

3. Import Indox modules and set the OpenAI embedding model:

   ```python
   from indox import IndoxRetrievalAugmentation
   from indox.embeddings import OpenAiEmbedding

   Indox = IndoxRetrievalAugmentation()
   openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
   ```

## Using IndoxApi Embedding Model

To use the IndoxApi embedding model, follow these steps:

1. Import necessary libraries and load environment variables:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   INDOX_API_KEY = os.environ['INDOX_API_KEY']
   ```

2. Import Indox modules and set the IndoxApi embedding model:

   ```python
   from indox import IndoxRetrievalAugmentation
   from indox.embeddings import IndoxApiEmbedding

   Indox = IndoxRetrievalAugmentation()
   embed_openai_indox = IndoxApiEmbedding(api_key=INDOX_API_KEY, model="text-embedding-3-small")
   ```

## Using Mistral Embedding Model

To use the Mistral embedding model, follow these steps:

1. Install the Mistral Python package:

   ```bash
   pip install mistralai
   ```

2. Load environment variables:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   MISTRAL_API_KEY = os.environ['MISTRAL_API_KEY']
   ```

3. Import Indox modules and set the Mistral embedding model:

   ```python
   from indox.embeddings import MistralEmbedding

   mistral_embedding = MistralEmbedding(api_key=MISTRAL_API_KEY)
   ```

## Using Hugging Face Embedding Model

To use the Hugging Face embedding model, follow these steps:

1. Load environment variables:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
   ```

2. Import Indox modules and set the Hugging Face embedding model:

   ```python
   from indox.embeddings import HuggingFaceEmbedding

   embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1")
   ```

## Using Clarifai Embedding Model

To use the Clarifai embedding model, follow these steps:

1. Install the Clarifai Python package:

   ```bash
   pip install clarifai
   ```

2. Load environment variables and import the Indox modules:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   CLARIFAI_PAT = os.environ['CLARIFAI_PAT']

   from indox.embeddings import ClarifaiEmbeddings

   clarifai_embeddings = ClarifaiEmbeddings(pat=CLARIFAI_PAT, model_id="model-id")
   ```

## Using Cohere Embedding Model

To use the Cohere embedding model, follow these steps:

1. Install the Cohere Python package:

   ```bash
   pip install cohere
   ```

2. Load environment variables and import the Indox modules:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   COHERE_API_KEY = os.environ['COHERE_API_KEY']

   from indox.embeddings import CohereEmbeddings

   cohere_embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
   ```

## Using Elasticsearch Embedding Model

To use the Elasticsearch embedding model, follow these steps:

1. Ensure you have an embedding model loaded and deployed in your Elasticsearch cluster.

2. Import necessary libraries and set the Elasticsearch embedding model:

   ```python
   from elasticsearch import Elasticsearch
   from indox.embeddings import ElasticsearchEmbeddings

   es_client = Elasticsearch("http://localhost:9200")
   es_embeddings = ElasticsearchEmbeddings(client=es_client, model_id="model-id")
   ```

## Using GPT4All Embedding Model

To use the GPT4All embedding model, follow these steps:

1. Install the GPT4All Python package:

   ```bash
   pip install gpt4all
   ```

2. Import Indox modules and set the GPT4All embedding model:

   ```python
   from indox.embeddings import GPT4AllEmbeddings

   gpt4all_embeddings = GPT4AllEmbeddings(model_name="gpt4all-lora")
   ```

## Using Ollama Embedding Model

To use the Ollama embedding model, follow these steps:

1. Follow the instructions at [Ollama](https://ollama.ai/) to set up the model locally.

2. Import Indox modules and set the Ollama embedding model:

   ```python
   from indox.embeddings import OllamaEmbeddings

   ollama_embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
   ```
   
## Using Azure OpenAi Embedding Model

To use the Azure OpenAI embedding model, follow these steps:

1. Install the OpenAI Python package:

   ```bash
   pip install openai
   ```

2. Import necessary libraries and load environment variables:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
   ```

3. Import Indox modules and set the Azure OpenAI embedding model:

   ```python
   from indox import IndoxRetrievalAugmentation
   from indox.embeddings import AzureOpenAIEmbeddings

   Indox = IndoxRetrievalAugmentation()
   openai_embeddings = AzureOpenAIEmbeddings(api_key=OPENAI_API_KEY,model="text-embedding-3-small")
   ```


### Future Plans

We are committed to continuously improving Indox and will be adding support for more embedding models in the future.
