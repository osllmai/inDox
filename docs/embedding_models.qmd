# Embedding Models


Indox currently supports two different embedding models. We plan to
increase the number of supported models in the future. The two supported
models are:

1.  **OpenAI Embedding Model**
2.  **Hugging Face Embedding Models**

## Using OpenAI Embedding Model

To use the OpenAI embedding model, follow these steps:

First install the OpenAI Python package:
```python
pip install openai
```

1.  Import necessary libraries and load environment variables:

``` python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

1.  Import Indox modules and set the OpenAI embedding model:

``` python
from indox import IndoxRetrievalAugmentation
from indox.embeddings import OpenAiEmbedding

Indox = IndoxRetrievalAugmentation()
openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
```

## Using Mistral Embedding Model

To use the Mistral embedding model, follow these steps:

First install the Mistral Python package:
```python
pip install mistralai
```

1.  Load environment variables:

``` python
import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.environ['MISTRAL_API_KEY']
```

2.  Import Indox modules and set the Mistral embedding model:

``` python
from indox.embeddings import MistralEmbedding

mistral_embedding = MistralEmbedding(api_key=MISTRAL_API_KEY)
```

### Future Plans

We are committed to continuously improving Indox and will be adding
support for more embedding models in the future.
