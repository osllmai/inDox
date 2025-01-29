---

## Deeplake vector store

### Overview

The `Deeplake` class interfaces with a DeepLake-based vector store for storing and searching text data using embeddings. This class provides methods to add documents to the vector store and perform similarity searches.

### Attributes

- **collection_name (str):** The name of the collection to store in the vector store.
- **embedding_function (callable):** A function that takes text input and returns an embedding vector.



### Example Usage

```python
# Import necessary classes from Indox library
from indoxArcg.llms import NerdToken
from indoxArcg.embeddings import NerdTokenEmbedding
from indoxArcg.data_loader_splitter import ClusteredSplit
from indoxArcg.vector_stores import Deeplake

# Create instances for API access and text embedding
openai_qa_indoxArcg = NerdToken(api_key="NERD TOKEN API KEY")
embed_openai_indoxArcg = NerdTokenEmbedding(api_key="NERD TOKEN API KEY", model="text-embedding-3-small")

# Specify the path to your text file
file_path = "sample.txt"

# Create a ClusteredSplit instance for handling file loading and chunking
loader_splitter = ClusteredSplit(file_path=file_path, embeddings=embed_openai_indoxArcg, summary_model=openai_qa_indoxArcg)

# Load and split the document into chunks using ClusteredSplit
docs = loader_splitter.load_and_chunk()

# Initialize the Deeplake instance
collection_name = "sample"
db = Deeplake(collection_name=collection_name, embedding_function=embed_openai_indox)

# Add documents to the vector store
db.add(docs=docs)

# Perform a similarity search
query = "example query text"
results = db.similarity_search_with_score(query=query, k=5)
```

---
