# Vector Store Integration in Indox

## Overview

Indox supports three vector stores for document retrieval: Postgres using pgvector, Chroma, and Faiss. This section provides an overview of the base vector store class and detailed instructions for configuring and using each supported vector store.

## Base Vector Store

The base vector store class `VectorStoreBase` defines the interface for vector-based document stores. It includes abstract methods for adding documents and retrieving documents similar to a given query.

### Class Definition

```python
class VectorStoreBase(ABC):
    """
    Abstract base class defining the interface for vector-based document stores.

    Methods:
        add_document: Abstract method to add documents to the vector store.
        retrieve: Abstract method to retrieve documents similar to the given query from the vector store.
    """

    @abstractmethod
    def add_document(self, docs):
        """
        Add documents to the vector store.

        Args:
            docs: The documents to be added to the vector store.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve documents similar to the given query from the vector store.

        Args:
            query (str): The query to retrieve similar documents.
            top_k (int, optional): The number of top similar documents to retrieve. Defaults to 5.
        """
        pass
```

## Configuration

Users should configure their vector store in a YAML file or set it directly in the code.

### YAML Configuration Example

```yaml
vector_store: "pgvector"
pgvector:
  conn_string: "postgresql+psycopg2://postgres:xxx@localhost:port/db_name"
```

### Code Configuration Example

```python
# Example of modifying a configuration setting
Indox.config["vector_store"] = "new_config"

# Applying the updated configuration
Indox.update_config()
```

## Postgres Using pgvector

To use pgvector as the vector store, users need to install pgvector and set the database address.

### Installation

For instructions on installing pgvector, refer to the pgvector installation guide.

### Configuration

Set the database connection string in the YAML file or in the code:

```yaml
pgvector:
  conn_string: "postgresql+psycopg2://postgres:xxx@localhost:port/db_name"
```

## Usage

Connect to the vector store:

```python
Indox.connect_to_vectorstore(collection_name="sample", embeddings=openai_embeddings)
```

Store documents in the vector store:

```python
Indox.store_in_vectorstore(chunks=docs)
```

Query the vector store:

```python
query = "your query?"
response = Indox.answer_question(query=query, qa_model=openai_qa)
```
