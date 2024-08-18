# Faiss

To use Faiss as the vector store, users need to install faiss and
set the embedding model.

```python
pip install faiss-cpu
```


### Hyperparameters
- embedding (Embedding): The embedding to be used.

### Installation

For instructions on installing faiss, refer to the FAISS
installation guide.

``` python
from indox.vector_stores import FAISSVectorStore
db = FAISSVectorStore(embedding=embed)
```

## Usage

Connect to the vector store:

``` python
Indox.connect_to_vectorstore(db)
```

Store documents in the vector store:

``` python
Indox.store_in_vectorstore(docs=docs)
```
