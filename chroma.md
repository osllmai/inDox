# Chroma

### Chroma

To use chroma as the vector store, users need to install Chroma and set the collection\_name of database and embedding model.

#### Hyperparameters

* collection\_name (str): The name of the collection in the database.
* embedding (Embedding): The embedding to be used.

#### Installation

For instructions on installing chroma, refer to the chroma installation guide.

```python
from indox.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="name",embedding=embed)
```

### Usage

Connect to the vector store:

```python
Indox.connect_to_vectorstore(db)
```

Store documents in the vector store:

```python
Indox.store_in_vectorstore(docs=docs)
```
