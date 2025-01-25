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

```python
from indoxArcg.vector_stores import FAISSVectorStore
db = FAISSVectorStore(embedding=embed)
```

## Usage

Store documents in the vector store:

```python
db.add(docs=docs)
```

```python
query = "How cinderella reach her happy ending?"
retriever = indoxArcg.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5, document_relevancy_filter=True)
answer = retriever.invoke(query=query)
```
