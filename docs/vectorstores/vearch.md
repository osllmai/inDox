# Vearch vector store

The `Vearch` class is designed to work with an external embedding function for creating a vector-based search database using the Vearch engine. It allows users to store documents, generate embeddings, and perform similarity searches.

### Initialization

```python
Vearch(embedding_function: Any, db_name: str, space_name: str = "text_embedding_space")
```

Initializes a Vearch instance with the following parameters:

- **embedding_function**: A function or model that converts text into embeddings. This function should have an `embed_query` method that returns a vector for a given query.
- **db_name**: The name of the database where the documents will be stored.
- **space_name**: The name of the vector space (default: `"text_embedding_space"`). This defines the structure of the data space in the database.

Upon initialization, the class automatically defines the dimensions of the embedding space by testing the embedding function with a sample query. It also creates a schema for the space.

### Attributes

- **embedding_function**: The external model or function used to create text embeddings.
- **db_name**: Name of the database.
- **space_name**: Name of the space.
- **data**: List of documents with their embeddings, initialized as an empty list.
- **dim**: The dimensionality of the embeddings, which is determined by the embedding function.
- **space_schema**: The schema for the space, including the fields for storing the text and embeddings.
- **config**: Configuration object to connect to the Vearch server, including host and token.

---

### Methods

#### `create_space_schema() -> SpaceSchema`
Creates a schema for the space with two fields: a document text field and an embedding field.

- **Returns**: A `SpaceSchema` object that defines the structure of the space.

#### `create_space(space_name: str)`
Creates a new space in the database using the defined schema.

- **space_name**: The name of the space to be created.
- **Returns**: The result of the space creation operation.

#### `create_database(db_name: str)`
Creates a new database.

- **db_name**: The name of the database to be created.
- **Returns**: The result of the database creation operation.

#### `add(docs: List[Document])`
Adds a list of documents into the Vearch database.

- **docs**: A list of `Document` objects. Each `Document` contains text and optionally metadata. The method computes the embeddings for each document using the `embedding_function` and appends them to the data storage.

#### `similarity_search_with_score(query: str, k: int = 5) -> List[Tuple[Document, float]]`
Performs a similarity search using a query string. It compares the embeddings of the query with the stored documents and returns the top `k` most similar documents along with their similarity scores.

- **query**: The text query to search for.
- **k**: The number of top results to return (default: 5).
- **Returns**: A list of tuples, where each tuple contains a `Document` object and a similarity score. The results are sorted in descending order of similarity.

---

### Example Usage

```python
openai_qa_indox = IndoxApi(api_key=INDOX_API_KEY)
embed_openai_indox = IndoxApiEmbedding(api_key=INDOX_API_KEY, model="text-embedding-3-small")

# Create a Vearch instance with the embedding function and a database name
db = Vearch(embedding_function=embed_openai_indox, db_name="sadra")

# Add documents to the database
db.add(docs=docs)

# Perform a similarity search
results = db.similarity_search_with_score(query="example search query", k=5)

# Output the top results
for doc, score in results:
    print(f"Document: {doc.page_content}, Similarity: {score}")
```

---

### Dependencies

- `vearch.config.Config`: Used to set up configuration for the Vearch client.
- `vearch.schema.field.Field`: Defines fields for the schema.
- `vearch.schema.space.SpaceSchema`: Defines the schema for the space in the Vearch database.
- `vearch.utils.DataType`, `vearch.utils.MetricType`: Utilities for defining the data types and metrics used for embedding similarity.
- `vearch.schema.index.FlatIndex`, `vearch.schema.index.ScalarIndex`: Indexing methods for fields.
- `sklearn.metrics.pairwise.cosine_similarity`: Used for computing the similarity between query embeddings and stored document embeddings.
