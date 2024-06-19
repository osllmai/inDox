# Embedding\_Models

Indox currently supports two different embedding models. We plan to increase the number of supported models in the future. The two supported models are:

1. **OpenAI Embedding Model**
2. **Hugging Face Embedding Models**

### Using OpenAI Embedding Model

To use the OpenAI embedding model, follow these steps:

1. Import necessary libraries and load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

1. Import Indox modules and set the OpenAI embedding model:

```python
from indox import IndoxRetrievalAugmentation
from indox.embeddings import OpenAiEmbedding

Indox = IndoxRetrievalAugmentation()
openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
```

### Using Hugging Face Embedding Model

To use the Hugging Face embedding model, follow these steps:

1. Import Indox modules and set the Hugging Face embedding model:

```python
from indox.embeddings import HuggingFaceEmbedding

hugging_face_embedding = HuggingFaceEmbedding(model_name="multi-qa-mpnet-base-cos-v1")
```

#### Future Plans

We are committed to continuously improving Indox and will be adding support for more embedding models in the future.

***

Previous: Unstructured Load and Split | Next: Question Answer Models
