# Embedding Models

indoxArcg supports multiple state-of-the-art embedding models for text representation. This guide provides detailed instructions for configuring and using each supported model.

## Table of Contents

- [Embedding Models](#embedding-models)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Supported Models](#supported-models)
  - [Model Configuration Guides](#model-configuration-guides)
    - [1. OpenAI Embedding Model](#1-openai-embedding-model)
    - [2. Azure OpenAI Embedding Model](#2-azure-openai-embedding-model)
    - [3. Hugging Face Embedding Models](#3-hugging-face-embedding-models)
    - [4. NerdToken Embedding Models](#4-nerdtoken-embedding-models)
    - [5. Mistral Embedding Model](#5-mistral-embedding-model)
    - [6. Clarifai Embedding Model](#6-clarifai-embedding-model)
    - [7. Cohere Embedding Model](#7-cohere-embedding-model)
    - [8. Elasticsearch Embedding Model](#8-elasticsearch-embedding-model)
    - [9. GPT4All Embedding Model](#9-gpt4all-embedding-model)
    - [10. Ollama Embedding Model](#10-ollama-embedding-model)
  - [Future Development](#future-development)

## Prerequisites

Before using any embedding models:

1. Python 3.8+ installed
2. Install required packages:
   ```bash
   pip install python-dotenv
   ```

Create a `.env` file in your project root with relevant API keys:

```ini
OPENAI_API_KEY=your_key_here
NERDTOKEN_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
# Add other API keys as needed
```

## Supported Models

| #   | Model Provider | Class Name              | Requirements                      |
| --- | -------------- | ----------------------- | --------------------------------- |
| 1   | OpenAI         | OpenAiEmbedding         | pip install openai                |
| 2   | Azure OpenAI   | AzureOpenAIEmbeddings   | pip install openai                |
| 3   | Hugging Face   | HuggingFaceEmbedding    | pip install sentence-transformers |
| 4   | NerdToken      | NerdTokenEmbedding      | API key required                  |
| 5   | Mistral        | MistralEmbedding        | pip install mistralai             |
| 6   | Clarifai       | ClarifaiEmbeddings      | pip install clarifai              |
| 7   | Cohere         | CohereEmbeddings        | pip install cohere                |
| 8   | Elasticsearch  | ElasticsearchEmbeddings | Elasticsearch instance running    |
| 9   | GPT4All        | GPT4AllEmbeddings       | pip install gpt4all               |
| 10  | Ollama         | OllamaEmbeddings        | Local Ollama server running       |

## Model Configuration Guides

### 1. OpenAI Embedding Model

Recommended Models: text-embedding-3-small, text-embedding-3-large

```python
import os
from dotenv import load_dotenv
from indoxArcg.embeddings import OpenAiEmbedding

load_dotenv()

embeddings = OpenAiEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv('OPENAI_API_KEY') # Required
)
```

### 2. Azure OpenAI Embedding Model

Required Parameters:

- azure_endpoint: Your Azure deployment endpoint
- deployment: Your model deployment name
- api_version: API version (e.g., "2023-05-15")

```python
from indoxArcg.embeddings import AzureOpenAIEmbeddings

azure_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://your-resource.openai.azure.com",
    deployment="your-deployment-name",
    api_version="2023-05-15",
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    model="text-embedding-3-small"
)
```

### 3. Hugging Face Embedding Models

Popular Models:

- sentence-transformers/all-MiniLM-L6-v2
- multi-qa-mpnet-base-cos-v1

```bash
pip install sentence-transformers
```

```python
from indoxArcg.embeddings import HuggingFaceEmbedding

hf_embeddings = HuggingFaceEmbedding(
    model="sentence-transformers/all-MiniLM-L6-v2",
    api_key=os.getenv('HUGGING_FACE_API_KEY')  # Optional
)
```

### 4. NerdToken Embedding Models

```python
from indoxArcg.embeddings import NerdTokenEmbedding

nt_embeddings = NerdTokenEmbedding(
    api_key=os.getenv('NERDTOKEN_API_KEY'),
    model="text-embedding-3-small",
)
```

### 5. Mistral Embedding Model

Current Model: mistral-embed (1280-dimension embeddings)

```python
from indoxArcg.embeddings import MistralEmbedding

mistral_embeddings = MistralEmbedding(
    api_key=os.getenv('MISTRAL_API_KEY'),
    model="mistral-embed",
)
```

### 6. Clarifai Embedding Model

Finding Model IDs:

1. Log in to Clarifai Portal
2. Navigate to your application
3. Copy model ID from model details

```python
from indoxArcg.embeddings import ClarifaiEmbeddings

clarifai_embeddings = ClarifaiEmbeddings(
    pat=os.getenv('CLARIFAI_PAT'),
    model_id="your-clarifai-model-id",
    user_id="clarifai-user-id",
    app_id="your-application-id"
)
```

### 7. Cohere Embedding Model

Recommended Models: embed-english-v3.0, embed-multilingual-v3.0

```python
from indoxArcg.embeddings import CohereEmbeddings

cohere_embeddings = CohereEmbeddings(
    api_key=os.getenv('COHERE_API_KEY'),
    model_name="embed-english-v3.0",
)
```

### 8. Elasticsearch Embedding Model

Prerequisites:

- Running Elasticsearch cluster (version 8.0+)
- Deployed embedding model via Eland

```bash
pip install elasticsearch
```

```python
from elasticsearch import Elasticsearch
from indoxArcg.embeddings import ElasticsearchEmbeddings

es = Elasticsearch(
    hosts=["http://localhost:9200"],
    basic_auth=("username", "password")
)

es_embeddings = ElasticsearchEmbeddings(
    client=es,
    model_id=".multilingual-e5-small"  # Example model ID
)
```

### 9. GPT4All Embedding Model

Available Models:

- all-MiniLM-L6-v2
- gpt4all-lora

```python
from indoxArcg.embeddings import GPT4AllEmbeddings

gpt4all_embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2",
    device="nvidia"  # or "cpu", "amd", "intel"
)
```

### 10. Ollama Embedding Model

Setup:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2
```

```python
from indoxArcg.embeddings import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="llama2",
    temperature=0.3  # Control randomness
)
```

## Future Development

Planned enhancements include:

- Integration with Google Vertex AI embedding models
- Support for multimodal embeddings (image+text)
- Batch embedding generation optimizations
- Dynamic model selection based on content type
- Enhanced error handling and retry mechanisms

For feature requests or issues, please visit our GitHub repository.
