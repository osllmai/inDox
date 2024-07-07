# Embedding Models


Indox currently supports two different embedding models. We plan to
increase the number of supported models in the future. The two supported
models are:

1.  **OpenAI Embedding Model**
2.  **Hugging Face Embedding Models**
3.  **IndoxApi Embedding Models**
4.  **Mistral Embedding Models**

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

2.  Import Indox modules and set the OpenAI embedding model:

``` python
from indox import IndoxRetrievalAugmentation
from indox.embeddings import OpenAiEmbedding

Indox = IndoxRetrievalAugmentation()
openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
```

## Using IndoxApi Embedding Model

To use the IndoxApi embedding model, follow these steps:


1.  Import necessary libraries and load environment variables:

``` python
import os
from dotenv import load_dotenv

load_dotenv()

INDOX_API_KEY = os.environ['INDOX_API_KEY']
```

2.  Import Indox modules and set the OpenAI embedding model:

``` python
from indox import IndoxRetrievalAugmentation
from indox.embeddings import IndoxApiEmbedding

Indox = IndoxRetrievalAugmentation()
embed_openai_indox = IndoxApiEmbedding(api_key=INDOX_API_KEY, model="text-embedding-3-small")
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

## Using HuggingFace Embedding Model

To use the Mistral embedding model, follow these steps:

1.  Load environment variables:

``` python
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
```

2.  Import Indox modules and set the Mistral embedding model:

``` python
from indox.embeddings import HuggingFaceEmbedding

embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1")
```


### Future Plans

We are committed to continuously improving Indox and will be adding
support for more embedding models in the future.
