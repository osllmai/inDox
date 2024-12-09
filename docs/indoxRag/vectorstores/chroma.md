# Chroma

To use chroma as the vector store, users need to install Chroma and
set the collection_name of database and embedding model.

```python
pip install chromadb
```

### Hyperparameters

- collection_name (str): The name of the collection in the database.
- embedding (Embedding): The embedding to be used.

### Installation

For instructions on installing chroma, refer to the chroma
installation guide.

```python
from indoxRag.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="name",embedding=embed)
```

## Usage

Store documents in the vector store:

```python
db.add(docs=docs)
```

```python
query = "How cinderella reach her happy ending?"
retriever = indoxRag.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5, document_relevancy_filter=True)
answer = retriever.invoke(query=query)
```
