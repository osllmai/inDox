[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/indoxArcg/blob/master/Demo/quick_start.ipynb)

# Quick Start

## Overview

This documentation provides a detailed explanation of how to use the
`IndoxRetrievalAugmentation` package for QA model and embedding
selection, document splitting, and storing in a vector store.

## Setup

### Install the Required Packages

```python
!pip install indoxArcg
!pip install openai
!pip install chromadb
```

### Load Environment Variables

To start, you need to load your API keys from the environment.

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

## Import indoxArcg Package

Import the necessary classes from the indoxArcg package.

```python
from indoxArcg.pipelines.rag import RAG
```

### Importing LLM and Embedding Models

```python
from indoxArcg.llms import OpenAi
```

```python
from indoxArcg.embeddings import OpenAiEmbedding
```

### Initialize indoxArcg

```python
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-0125")
openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small",openai_api_key=OPENAI_API_KEY)
```

```python
file_path = "sample.txt"
```

In this section, we take advantage of the `unstructured` library to load
documents and split them into chunks by title. This method helps in
organizing the document into manageable sections for further
processing.

```python
from indoxArcg.data_loader_splitter import UnstructuredLoadAndSplit
```

```python
loader_splitter = UnstructuredLoadAndSplit(file_path=file_path)
docs = loader_splitter.load_and_chunk()
```

    Starting processing...
    End Chunking process.

Storing document chunks in a vector store is crucial for enabling
efficient retrieval and search operations. By converting text data into
vector representations and storing them in a vector store, you can
perform rapid similarity searches and other vector-based operations.

```python
from indoxArcg.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed_openai)
indoxArcg.connect_to_vectorstore(db)
indoxArcg.store_in_vectorstore(docs)
```

    2024-05-14 15:33:04,916 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
    2024-05-14 15:33:12,587 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    2024-05-14 15:33:13,574 - INFO - Document added successfully to the vector store.

    Connection established successfully.

    <indoxArcg.vectorstore.ChromaVectorStore at 0x28cf9369af0>

## Quering

```python
query = "how cinderella reach her happy ending?"
```

```python
retriever = RAG(vector_database=db,llm=openai_qa)
retriever.infer(query,,top_k=5)
```

    2024-05-14 15:34:55,380 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
    2024-05-14 15:35:01,917 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    'Cinderella reached her happy ending by enduring mistreatment from her step-family, finding solace and help from the hazel tree and the little white bird, attending the royal festival where the prince recognized her as the true bride, and ultimately fitting into the golden shoe that proved her identity. This led to her marrying the prince and living happily ever after.'

```python

```
