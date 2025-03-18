# Multi Vector Retriever

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

````python
from indoxArcg.vector_stores import Chroma, Milvus, ApacheCassandra, FAISS, PGVector, Deeplake
from indoxArcg.multi_vector_retriever import MultiVectorRetriever

# Define vector store instances
db1 = Deeplake(embedding_function=embed_openai_indoxArcg, collection_name="sample")
db2 = ApacheCassandra(embedding_function=embed_openai_indoxArcg, keyspace="sample")

# Add documents to the vector stores
db1.add(docs=docs1)
db2.add(docs=docs2)

# Initialize MultiVectorRetriever with multiple vector stores
multivector = MultiVectorRetriever(vector_stores=[db1, db2])

# Define the query
query = "How cinderella reach her happy ending?"

```python
from indoxArcg.pipelines.rag import RAG

retriever = RAG(llm,multivector)
answer = retriever.infer(
    question=query,
    top_k=5,
)
````
