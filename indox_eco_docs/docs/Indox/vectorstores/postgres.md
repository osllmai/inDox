# Postgres Using PgVector

First you should install pgvector and set the database address.
```python
pip install pgvector
pip install psycopg2
```
To use pgvector as the vector store, users need to install pgvector and
set the database address.

### Hyperparameters
- host (str): The host of the PostgreSQL database.
- port (int): The port of the PostgreSQL database.
- dbname (str): The name of the database.
- user (str): The user for the PostgreSQL database.
- password (str): The password for the PostgreSQL database.
- collection_name (str): The name of the collection in the database.
- embedding (Embedding): The embedding to be used.

### Installation

For instructions on installing pgvector, refer to the pgvector
installation guide.

``` python
from indox.vector_stores import PGVectorStore
db = PGVectorStore(host="host",port=port,dbname="dbname",user="username",password="password",collection_name="sample",embedding=embed)
```

## Usage

Store documents in the vector store:

``` python
db.add(docs=docs)
```

``` python
query = "How cinderella reach her happy ending?"
retriever = indox.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5, document_relevancy_filter=True)
answer = retriever.invoke(query=query)
```

