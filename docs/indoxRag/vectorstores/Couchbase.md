# Couchbase Vector Store

## Overview

This code defines a `Couchbase` class that integrates with Couchbase, allowing the storage and retrieval of documents using vector embeddings. It provides methods to add documents with their corresponding embeddings to a Couchbase bucket and perform similarity searches based on a query. The embedding function is used to convert the text into vectors for storage and querying.

The class utilizes the `indoxRag` library for embedding functions and document representation. The `similarity_search_with_score` method returns a ranked list of documents with their similarity scores, enabling a search functionality based on vector similarity.

## Requirements

- Couchbase server instance with a properly configured bucket.
- `couchbase` Python SDK (Cluster, Bucket, etc.)

## Class: `Couchbase`

### Constructor: `__init__`

```python
def __init__(self, embedding_function: Callable[[str], np.ndarray], bucket_name: str,
             cluster_url: str = 'couchbase://localhost', username: str = 'Administrator',
             password: str = 'indoxRagmain')
```

#### Parameters:

- **embedding_function** (`Callable[[str], np.ndarray]`): A function used to embed the documents and queries into vector representations.
- **bucket_name** (`str`): The name of the Couchbase bucket used to store the documents.
- **cluster_url** (`str`, optional): The URL of the Couchbase cluster (default: `couchbase://localhost`).
- **username** (`str`, optional): Username for Couchbase authentication (default: `Administrator`).
- **password** (`str`, optional): Password for Couchbase authentication (default: `indoxRagmain`).

#### Exceptions:

- **CouchbaseException**: Raised if there's an error in connecting to Couchbase.

### Method: `add`

```python
def add(self, docs: List[str])
```

#### Parameters:

- **docs** (`List[str]`): A list of document strings to be added to the Couchbase bucket.

#### Functionality:

- Each document in the list is embedded using the `embedding_function`, generating a vector representation.
- A unique document ID is generated using `uuid.uuid4()`.
- The document's content and its vector embedding are stored in Couchbase using the `upsert` method.

#### Exceptions:

- **ValueError**: Raised if the `docs` argument is not a list.
- **CouchbaseException**: Raised if there's an error while adding documents to Couchbase.

### Method: `similarity_search_with_score`

```python
def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]
```

#### Parameters:

- **query** (`str`): The query string for which similarity search is performed.
- **k** (`int`, optional): The number of top results to return (default: 5).

#### Returns:

- **List[Tuple[Document, float]]**: A list of tuples containing `Document` objects and their corresponding similarity scores, sorted in descending order of similarity.

#### Functionality:

- Embeds the query using the `embedding_function`.
- Searches for similar documents in the Couchbase bucket using full-text search and retrieves documents along with their similarity scores.
- Constructs `Document` objects with the retrieved content and metadata, such as document ID.
- The results are sorted by similarity score and returned as a list of `(Document, score)` tuples.

#### Exceptions:

- **CouchbaseException**: Raised if there's an error during the similarity search or document retrieval.

## Example Usage

```python
import numpy as np
from indoxRag.vector_stores import Couchbase

# Initialize the Couchbase instance with an embedding function
db = Couchbase(embedding_function=embed_openai_indoxRag, bucket_name="QA")

db.add(docs)

# Query the database for a similarity search
query = "Does Travelers Insurance Have Renters Insurance?"

# Use the Couchbase instance within the indoxRag.QuestionAnswer class
retriever = indoxRag.QuestionAnswer(vector_database=db, llm=openai_qa_indoxRag, top_k=5)

# Invoke the retriever with the query
retriever.invoke(query)
```

### Explanation:

1. **Embedding Function**: The Couchbase class uses an embedding function (e.g., `embed_openai_indoxRag`) to convert text into vector embeddings before storing or querying.
2. **Querying for Similarity**: The `similarity_search_with_score` method is called within the `indoxRag.QuestionAnswer` class, enabling a hybrid retrieval system that returns documents based on vector similarity.
3. **Retrieval Process**: The `retriever.invoke(query)` is responsible for executing the query and retrieving the relevant results. It utilizes the Couchbase class for vector database interaction and the `openai_qa_indoxRag` LLM for question answering.

### Output:

The retrieval process returns the most similar documents to the query, along with their relevance scores, allowing for efficient similarity-based search.

## Notes:

- Ensure that the Couchbase Full-Text Search (FTS) index `FTS_QA` is properly set up for querying.
- Customize the `embedding_function` to match the vectorization model used in your project.
