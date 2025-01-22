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
from indoxArcg.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="name",embedding=embed)
```
