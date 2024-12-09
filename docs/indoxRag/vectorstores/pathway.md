# Pathway Vector Store

To use Pathway Vector Store, you can create a client instance by specifying the `host`, `port`, or `url`.

### Hyperparameters

- `host` (Optional[str]): The host on which Pathway Vector Store listens.
- `port` (Optional[int]): The port on which Pathway Vector Store listens.
- `url` (Optional[str]): The URL at which Pathway Vector Store listens.

### Installation

You can connect to the Pathway Vector Store by creating an instance of the `PathwayVectorClient`:

```python
from indoxRag.vector_stores import PathwayVectorClient
client = PathwayVectorClient(host="localhost", port=8080)
```

Alternatively, you can connect using a full URL:

```python
client = PathwayVectorClient(url="http://localhost:8080")
```

## Usage

Once the client is initialized, you can use it to query the Pathway Vector Store.

Store documents in the vector store:

```python
client.add(docs=docs)
```

Query the vector store:

```python
query = "How can Cinderella reach her happy ending?"
retriever = indoxRag.QuestionAnswer(vector_database=client, llm=mistral_qa, top_k=5, document_relevancy_filter=True)
answer = retriever.invoke(query=query)
```
