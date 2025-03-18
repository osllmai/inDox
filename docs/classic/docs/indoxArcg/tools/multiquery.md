# Multi Query Retrieval

## Overview

The `MultiQueryRetrieval` class implements an advanced information retrieval technique that generates multiple queries from an original query, retrieves relevant information for each generated query, and combines the results to produce a comprehensive final response.

## Class: MultiQueryRetrieval

### Initialization

```python
def __init__(self, llm, vector_database, top_k: int = 3):
```

- **llm**: The language model used for query generation and response synthesis.
- **vector_database**: The vector database used for information retrieval.
- **top_k**: The number of top results to retrieve for each query (default: 3).

### Methods

**generate_queries**

```python
def generate_queries(self, original_query: str) -> List[str]:
```

**Generates multiple queries from the original query.**

- **original_query:** The original user query.
- **Returns:** A list of generated queries.

**retrieve_information**

```python
def retrieve_information(self, queries: List[str]) -> List[str]:
```

**Retrieves relevant information for each generated query.**

- **queries:** A list of queries to use for information retrieval.
- **Returns:** A list of relevant passages retrieved from the vector database.

**generate_response**

```python
def generate_response(self, original_query: str, context: List[str]) -> str:
```

**Generates a final response based on the original query and retrieved context.**

- **original_query:** The original user query.
- **context:** A list of relevant passages to use as context.
- **Returns:** The generated response.

**run**

```python
def run(self, query: str) -> str:
```

**Executes the full multi-query retrieval process.**

- **query:** The original user query.
- **Returns:** The final generated response.

## Usage

To use the MultiQueryRetrieval functionality within the `RAG` class:

```python
from indoxArcg.pipelines.rag import RAG

retriever = RAG(llm,vector_store)
answer = retriever.infer(
    question=query,
    top_k=5,
    use_clustering=False,
    use_multi_query=True
)
```
