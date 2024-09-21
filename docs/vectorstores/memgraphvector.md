# MemgraphVector

The `MemgraphVector` class allows you to use Memgraph for similarity search on graph-structured data by leveraging embeddings. This can be used for tasks such as information retrieval and recommendation systems where vector or keyword search is needed. The class supports different search methods: vector-based, keyword-based, and hybrid search.

# Hyperparameters

- **uri** (str): The URI of the Memgraph instance (e.g., `bolt://localhost:7687`).
- **username** (str): The username for authenticating to the Memgraph database.
- **password** (str): The password for authenticating to the Memgraph database.
- **embedding_function** (Callable): A function to convert query text into embeddings (e.g., OpenAI model, custom embedding function).
- **search_type** (str, optional): The default type of search ('vector', 'keyword', or 'hybrid'). Defaults to `'vector'`.

# Installation

To use `MemgraphVector`, you need to install the `neo4j` driver and ensure you have a running Memgraph instance. You can install the required driver with:

```bash
pip install neo4j
```

Refer to the official Memgraph documentation for setting up a Memgraph instance.

# Example Usage
## Initialize MemgraphVector

You can initialize the `MemgraphVector` class by passing the necessary connection details and the embedding function.

```python
from indox.vector_stores import MemgraphVector

# Initialize MemgraphVector with connection details and an embedding function
memgraph_vector = MemgraphVector(
    uri="bolt://localhost:7687",
    username="username",
    password="password",
    embedding_function=my_embedding_function,  # Your embedding function
    search_type='vector'
)
```

## Perform a Similarity Search
To perform a similarity search, use the `search` method, which supports vector, keyword, or hybrid search types.

```python
# Run a similarity search
results = memgraph_vector.search(query="Sample query text", search_type="vector", k=5)

# Print the results
for document in results:
    print(document.page_content)
```

- **query** (str): The search query text.
- **search_type** (str, optional): The type of search to perform ('vector', 'keyword', or 'hybrid'). Defaults to the class-level search type.
- **k** (int, optional): The number of top results to return. Defaults to 4.

## Retrieve Documents with Scores
You can also retrieve documents along with their similarity scores using the `retrieve` method.

```python
# Retrieve documents with similarity scores
contexts, scores = memgraph_vector.retrieve(query="Sample query", top_k=5, search_type="hybrid")

# Output the documents and their scores
for context, score in zip(contexts, scores):
    print(f"Document: {context}, Score: {score}")
```

- **top_k** (int, optional): The number of top results to return. Defaults to 5.
- **search_type** (str, optional): The type of search to perform ('vector', 'keyword', or 'hybrid'). Defaults to the class-level search type.

## Close the Connection
After performing your operations, remember to close the Memgraph connection:

```python
memgraph_vector.close()
```

# Search Methods
The `MemgraphVector` class supports the following search methods:

- **_run_vector_search**: Searches for similar embeddings based on cosine similarity.
- **_run_keyword_search**: Searches for documents containing the keyword query.
- **_run_hybrid_search**: Combines vector and keyword search with weighted scores.

# Example Workflow

```python
from indox.vector_stores import MemgraphVector

# Initialize MemgraphVector
memgraph_vector = MemgraphVector(
    uri="bolt://localhost:7687",
    username="username",
    password="password",
    embedding_function=my_embedding_function,
    search_type="vector"
)

# Perform vector similarity search
vector_results = memgraph_vector.search(query="Machine learning", search_type="vector", k=5)

# Perform hybrid search
hybrid_results = memgraph_vector.search(query="Artificial intelligence", search_type="hybrid", k=5)

# Close the Memgraph connection
memgraph_vector.close()
```
