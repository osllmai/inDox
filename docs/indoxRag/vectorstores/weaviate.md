# Weaviate

To use Weaviate as the vector store, users need to install the `weaviate-client` and set the `index_name`, `text_key`, and embedding model.

```bash
pip install weaviate-client
```

### Hyperparameters

- `client` (Any): The Weaviate client instance.
- `index_name` (str): The name of the index in the Weaviate database.
- `text_key` (str): The key of the text field in the objects.
- `embedding` (Optional[Embeddings]): The embedding model to be used.
- `attributes` (Optional[List[str]]): Specific attributes to be retrieved.
- `relevance_score_fn` (Optional[Callable[[float], float]]): Function to normalize the relevance scores.
- `by_text` (bool): If True, embeddings are calculated by the text content.

### Installation

For instructions on installing Weaviate, refer to the Weaviate installation guide.

```python
from indoxRag.vector_stores import WeaviateVectorStore
db = WeaviateVectorStore(client=weaviate_client, index_name="index_name", text_key="text_key", embedding=embed)
```

## Usage

Store documents in the vector store:

```python
db.add(docs=docs)
```

Query the vector store:

```python
query = "How cinderella reach her happy ending?"
retriever = indoxRag.QuestionAnswer(vector_database=db, llm=mistral_qa, top_k=5, document_relevancy_filter=True)
answer = retriever.invoke(query=query)
```

## Running Weaviate in Docker

To run Weaviate in Docker, use the following commands:

Pull the latest Weaviate Docker image:

```bash
docker pull semitechnologies/weaviate:latest
```

Run Weaviate in a container:

```bash
docker run -d -p 8080:8080 semitechnologies/weaviate:latest
```

If you encounter permission errors, you can run Weaviate with the following command:

```bash
docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -v /path/on/host:/var/lib/weaviate semitechnologies/weaviate:latest
```
