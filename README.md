# inDox Retrieval Augmentation

## Overview

This project combines advanced clustering techniques provided by Raptor with the efficient retrieval capabilities of pgvector. It allows users to interact with and visualize their data in a PostgreSQL database. The solution involves segmenting text data into manageable chunks, enhancing retrieval through a custom model, and providing an interface for querying and retrieving relevant information.

## Prerequisites

Before you can run this project, you need the following installed:
- Python 3.8+
- PostgreSQL
- OpenAI API Key (for embedding models)

Ensure your system also meets the following requirements:
- Access to environmental variables for sensitive information (API keys).
- Suitable hardware to support intensive computational tasks.


## Installation

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/osllmai/inDox.git
cd inDox
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables
Set your `OPENAI_API_KEY` in your environment variables for secure access.

### Database Setup
Ensure your PostgreSQL database is up and running, and accessible from your application.

## Usage

### Preparing Your Data

1. **Define the File Path**: Specify the path to your text or PDF file.
2. **Load Embedding Models**: Initialize your embedding model from OpenAI's selection of pre-trained models.

## Clustering and Retrieval

### Initialize the Retrieval System

```python
from Indox import IndoxRetrievalAugmentation
IRA = IndoxRetrievalAugmentation(docs='path/to/your/file', embeddings='your_embedding_model', max_tokens=500)
```

### Generate Chunks

```python
all_chunks = IRA.get_all_chunks()
print("Chunks:", all_chunks)
```


### PostgreSQL Setup with pgvector

1. **Install pgvector**: To install `pgvector` on your PostgreSQL server, follow the detailed installation instructions available on the official pgvector GitHub repository:

[pgvector Installation Instructions](https://github.com/pgvector/pgvector)

2. **Add Vector Extension**:
   Connect to your PostgreSQL database and run the following SQL command to create the `pgvector` extension:

```sql
-- Connect to your database
psql -U username -d database_name

-- Run inside your psql terminal
CREATE EXTENSION vector;
# Replace the placeholders with your actual PostgreSQL credentials and details
```

### Store in PostgreSQL

```python
# Replace the placeholders with your actual PostgreSQL credentials and details
connection_string = "postgresql+psycopg2://username:password@hostname:port/database_name"
collection_name = "collection_name"

IRA.store_in_postgres(collection_name=collection_name,
                     connection_string=connection_string,
                     all_chunks=all_chunks)
```

In this snippet:
- **username**: Replace with your PostgreSQL username.
- **password**: Replace with your PostgreSQL password.
- **hostname**: Replace with the address of your PostgreSQL server (e.g., localhost).
- **port**: Replace with the port number your PostgreSQL server is running on (e.g., 5432).
- **database_name**: Replace with the name of your PostgreSQL database.
- **collection_name**: Replace with the name of the collection where you want to store the chunks.


### Querying

```python
response, scores = IRA.answer_question(query="your question here", top_k=5)
print("Responses:", response)
print("Scores:", scores)
```
## Roadmap

### Vectorstores
- [x] pgvector
- [x] chromadb
- [ ] lancedb
### Summary models
- [x] openai
- [x] huggingface models

### Embedding models
- [x] openai embeddings
- [ ] sentence transformer embeddings

### Chunking strategies
- [ ] semantic chunking

### Metrics
- [ ] add some metrics for measuring performance and quality of the system
      
### Other features
- [x] Add yaml file

