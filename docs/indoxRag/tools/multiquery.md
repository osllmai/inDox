# MultiQueryRetrieval Documentation

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

To use the MultiQueryRetrieval functionality within the `IndoxRetrievalAugmentation` class:

1. Initialize the IndoxRetrievalAugmentation class:

```python
ira = IndoxRetrievalAugmentation()
```

2. Create a `QuestionAnswer` instance:

```python
qa = ira.QuestionAnswer(llm=lm, vector_database=vector_db, top_k=5)
```

3. Use the `invoke` method to perform multi-query retrieval:

```python
response = qa.invoke("Your complex query here",multi_query=True)
```

### Example

```python
from indoxRag import IndoxRetrievalAugmentation

# Initialize the retrieval augmentation
ira = IndoxRetrievalAugmentation()

# Create a QuestionAnswer instance
qa = ira.QuestionAnswer(
    llm=llm,
    vector_database=vector_db,
    top_k=5
)

# Perform multi-query retrieval
query = "What are the main differences between renewable and non-renewable energy sources?"
response = qa.invoke(query,multi_query=True)

print(f"Query: {query}")
print(f"Response: {response}")

```
