# `MultiVectorRetriever`

## Overview

The `MultiVectorRetriever` class allows users to perform similarity searches across multiple vector stores in parallel and retrieve results based on their similarity score. This class is particularly useful for projects that need to integrate several vector stores and combine search results, such as building a multivector hybrid search system.

### Key Features

- Perform similarity searches on multiple vector stores simultaneously.
- Combine and sort the results from all vector stores based on similarity scores.
- Handle exceptions during the search process and log any issues.

### Constructor: `__init__(self, vector_stores: List[Any])`

This method initializes the `MultiVectorRetriever` object.

#### Parameters:

- `vector_stores` (List[Any]): A list of initialized vector store instances, each capable of performing similarity searches (e.g., Deeplake, ApacheCassandra).

### Method: `similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Any, float]]`

Executes similarity searches across all vector stores, returning the most relevant results based on their similarity scores.

#### Parameters:

- `query` (str): The query text to be searched across the vector stores.
- `k` (int): The number of results to return. Defaults to 5.

#### Returns:

- List[Tuple[Any, float]]: A list of tuples where each tuple contains a document and a similarity score. The list is sorted in descending order of similarity.

#### Raises:

- Logs any exceptions encountered during the search process.

### Example Usage

In this example, we initialize two vector stores (`Deeplake` and `ApacheCassandra`), add documents to them, and then create a `MultiVectorRetriever` instance. A query is then run across both vector stores, retrieving the top 5 most similar results.

```python
from indoxRag.vector_stores import Chroma, Milvus, ApacheCassandra, FAISS, PGVector, Deeplake
from indoxRag.multi_vector_retriever import MultiVectorRetriever

# Define vector store instances
db1 = Deeplake(embedding_function=embed_openai_indoxRag, collection_name="sample")
db2 = ApacheCassandra(embedding_function=embed_openai_indoxRag, keyspace="sample")

# Add documents to the vector stores
db1.add(docs=docs1)
db2.add(docs=docs2)

# Initialize MultiVectorRetriever with multiple vector stores
multivector = MultiVectorRetriever(vector_stores=[db1, db2])

# Define the query
query = "How cinderella reach her happy ending?"

# Perform the retrieval
retriever = indoxRag.QuestionAnswer(vector_database=multivector, llm=openai_qa_indoxRag, top_k=5)

# Example query results
results = retriever.retrieve(query=query)
```

### Explanation

1. **Vector Store Initialization**: We initialize two vector stores, `Deeplake` and `ApacheCassandra`. Each of these is configured with an embedding function and other parameters specific to the store.
2. **Adding Documents**: Documents (`docs1`, `docs2`) are added to the vector stores, allowing them to store vectors for future searches.

3. **MultiVectorRetriever**: A `MultiVectorRetriever` object is created, taking in a list of vector stores (`db1`, `db2`).

4. **Query**: The query string, "How cinderella reach her happy ending?", is used to search across all vector stores. The results are returned from the `QuestionAnswer` system, which uses the `MultiVectorRetriever` for similarity search and `openai_qa_indoxRag` as the language model.

5. **Retrieval**: The `retriever.retrieve()` method executes the query and retrieves the top 5 results based on their similarity scores.

### Logging

This class uses the `loguru` library to log the following:

- **INFO** level logs for general information.
- **ERROR** level logs for exceptions encountered during the search process.
