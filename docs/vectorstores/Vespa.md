# Vespa Vector store

## Overview

This code defines a `VESPA` class that interacts with a Vespa application to handle text embeddings, document storage, and semantic search functionality. The Vespa application uses a tensor-based embedding for semantic search and integrates with an embedding model. The documents are processed by extracting titles, generating embeddings, and storing them in Vespa. Users can query the stored documents using semantic similarity based on embeddings.

The class provides methods to deploy the Vespa application, add documents, retrieve relevant documents based on queries, and display the results in a pandas DataFrame. It also includes a `similarity_search_with_score` method to return top documents and their similarity scores.

## Key Components

- **extract_title(text: str)**: Extracts the first sentence from the input text to use as a title.
- **generate_id()**: Generates a unique identifier using UUID for each document.
- **process_docs(input_list: List[str], embedding_function: Any)**: Processes a list of text documents, generating titles, embeddings, and document IDs.
- **VESPA class**: Main class to interact with Vespa. It manages deployment, adding documents, querying, and similarity search.
  - `__init__(app_name: str, embedding_function: Any)`: Initializes the Vespa instance, configures the schema, and determines the embedding dimensions dynamically.
  - `deploy()`: Deploys the Vespa application using Docker.
  - `add(docs: List[Dict[str, Any]])`: Adds a list of documents to Vespa, embedding them before ingestion.
  - `_get_relevant_documents(query: str)`: Retrieves relevant documents for a given query using semantic search.
  - `display_hits_as_df(documents: List[Document])`: Displays the retrieved documents and their relevance as a pandas DataFrame.
  - `similarity_search_with_score(query: str, k: int)`: Performs a similarity search and returns the top `k` documents along with their scores.

## Example Usage

```python
# Initialize Vespa instance with an application name and embedding function
db = VESPA(app_name="xxxx", embedding_function=embed_openai_indox)

# Add the documents to Vespa, embedding each one
db.add(docs=docs)

# Perform a semantic search on Vespa for relevant documents based on a query
query = "How did Cinderella reach her happy ending?"
retriever = indox.QuestionAnswer(vector_database=db, llm=openai_qa_indox, top_k=5)

# Retrieve the most relevant context based on the query
context = retriever.context
print(context)
```

### Explanation

1. **Initializing Vespa**: `VESPA(app_name="xxxx", embedding_function=embed_openai_indox)` initializes a Vespa instance with the given application name and an embedding function (`embed_openai_indox`), which will handle generating embeddings for the documents.
   
2. **Adding Documents**: `db.add(docs=docs)` processes and adds a list of document texts to Vespa. Each document is embedded using the specified embedding function and stored in the Vespa schema.

3. **Querying**: The query `"How did Cinderella reach her happy ending?"` is processed using the `QuestionAnswer` retriever from the `indox` library. The retriever queries the Vespa instance (`db`) to retrieve relevant documents based on semantic similarity.

4. **Result Retrieval**: The `retriever.context` retrieves and displays the most relevant documents and their contexts for the query.
