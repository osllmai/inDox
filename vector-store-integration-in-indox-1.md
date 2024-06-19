# Vector Store Integration in Indox

## Overview

Indox supports three vector stores for document retrieval: Postgres using pgvector, Chroma, and Faiss. This section provides an overview of the base vector store class and detailed instructions for configuring and using each supported vector store.

## Postgres Using PgVector

To use pgvector as the vector store, users need to install pgvector and set the database address.

### Hyperparameters

* host (str): The host of the PostgreSQL database.
* port (int): The port of the PostgreSQL database.
* dbname (str): The name of the database.
* user (str): The user for the PostgreSQL database.
* password (str): The password for the PostgreSQL database.
* collection\_name (str): The name of the collection in the database.
* embedding (Embedding): The embedding to be used.

### Installation

For instructions on installing pgvector, refer to the pgvector installation guide.

```python
from indox.vector_stores import PGVectorStore
db = PGVectorStore(host="host",port=port,dbname="dbname",user="username",password="password",collection_name="sample",embedding=embed)
```

## Usage

Connect to the vector store:

```python
Indox.connect_to_vectorstore(db)
```

Store documents in the vector store:

```python
Indox.store_in_vectorstore(docs=docs)
```

## Chroma

To use chroma as the vector store, users need to install Chroma and set the collection\_name of database and embedding model.

### Hyperparameters

* collection\_name (str): The name of the collection in the database.
* embedding (Embedding): The embedding to be used.

### Installation

For instructions on installing chroma, refer to the chroma installation guide.

```python
from indox.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="name",embedding=embed)
```

## Usage

Connect to the vector store:

```python
Indox.connect_to_vectorstore(db)
```

Store documents in the vector store:

```python
Indox.store_in_vectorstore(docs=docs)
```

## Faiss

To use Faiss as the vector store, users need to install faiss and set the embedding model.

### Hyperparameters

* embedding (Embedding): The embedding to be used.

### Installation

For instructions on installing faiss, refer to the FAISS installation guide.

```python
from indox.vector_stores import FAISSVectorStore
db = FAISSVectorStore(embedding=embed)
```

## Usage

Connect to the vector store:

```python
Indox.connect_to_vectorstore(db)
```

Store documents in the vector store:

```python
Indox.store_in_vectorstore(docs=docs)
```
