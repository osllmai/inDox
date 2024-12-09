### `ApacheCassandra` Vector store

#### Overview

The `ApacheCassandra` class provides an interface to store, manage, and search through text data using embeddings, with the Apache Cassandra database as the backend. This class enables you to add documents with their embeddings to Cassandra and perform similarity searches to retrieve the most relevant documents.

#### Attributes

- **cluster_ips (List[str])**: List of Cassandra node IP addresses (currently hardcoded to `127.0.0.1`).
- **port (int)**: Port number to connect to Cassandra (default is `9042`).
- **embedding_function (Callable[[str], np.ndarray])**: A function that takes text input and returns an embedding vector as a NumPy array.
- **keyspace (str)**: The keyspace (database) in Cassandra to use for storage.

#### Methods

##### `__init__(self, embedding_function: Callable[[str], np.ndarray], keyspace: str)`

Initializes the `ApacheCassandra` instance.

- **Parameters**:

  - `embedding_function`: A callable that computes embedding vectors from text (used for both adding documents and queries).
  - `keyspace`: The Cassandra keyspace where the data is stored.

- **Raises**:
  - `RuntimeError`: If Cassandra connection initialization fails.

##### `_setup_keyspace(self)`

Private method to set up the Cassandra keyspace and create the required table for storing embeddings.

- **Raises**:
  - `RuntimeError`: If the keyspace or table setup fails.

##### `add(self, docs: List[str])`

Adds a list of documents to the Cassandra vector store by embedding the documents and storing the embeddings.

- **Parameters**:

  - `docs`: A list of text documents to be embedded and stored in the Cassandra database.

- **Raises**:
  - `ValueError`: If `docs` is not a list of strings.
  - `RuntimeError`: If there is any issue while adding the documents to the Cassandra store.

##### `similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]`

Performs a similarity search on the Cassandra vector store using the provided query. The method returns the most similar documents along with their cosine similarity scores.

- **Parameters**:

  - `query`: A string query text for which similar documents are to be found.
  - `k`: An integer specifying the number of top results to return (default: `5`).

- **Returns**:

  - A list of tuples where each tuple contains:
    - `Document`: The document object corresponding to the retrieved text.
    - `float`: The similarity score (cosine similarity) between the query and the document.

- **Raises**:
  - `RuntimeError`: If there is an issue with the Cassandra operation or similarity search.

##### `shutdown(self)`

Shuts down the Cassandra cluster connection.

- **Raises**:
  - `RuntimeError`: If Cassandra shutdown fails.

---

### Example Usage

```python
from indoxRag.vector_stores import ApacheCassandra
from indoxRag.embeddings import OpenAiEmbedding
from indoxRag.llms import OpenAi
from indoxRag.core import Document

# Initialize Apache Cassandra vector store with OpenAI embedding function
db = ApacheCassandra(embedding_function=embed_openai_indoxRag, keyspace="sample")

db.add(docs=docs)

# Perform a similarity search with a query
query = "How did Cinderella reach her happy ending?"
retriever = indoxRag.QuestionAnswer(vector_database=db, llm=openai_qa_indoxRag, top_k=5)

retriever.invoke(query)

# Shutdown the Cassandra connection
db.shutdown()
```

### Explanation:

1. **Initialization**: The `ApacheCassandra` instance is created by passing an embedding function (`embed_openai_indoxRag`) and specifying a keyspace (`"sample"`).
2. **Adding Documents**: A list of documents (`docs`) is embedded and stored in the Cassandra vector store via the `add` method.

3. **Similarity Search**: A query is provided (`"How did Cinderella reach her happy ending?"`), and a `QuestionAnswer` retriever is used to perform a similarity search with the top 5 most relevant documents.

4. **Retrieval**: After invoking the retriever with the query, the relevant context is extracted from the results and printed.

5. **Shutdown**: The Cassandra connection is gracefully closed using the `shutdown` method.
