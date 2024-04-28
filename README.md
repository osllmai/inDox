# inDox : Advance Search and Retrieval Augmentation Generative  

## Overview 

This project combines advanced clustering techniques provided by Raptor with the efficient retrieval capabilities of pgvector and other vectorstores. It allows users to interact with and visualize their data in a PostgreSQL database. The solution involves segmenting text data into manageable chunks, enhancing retrieval through a custom model, and providing an interface for querying and retrieving relevant information.

## Prerequisites

Before you can run this project, you need the following installed:
- Python 3.8+
- PostgreSQL (if you want to store your data on postgres)
- OpenAI API Key (if using OpenAI embedding model)

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
Ensure your PostgreSQL database is up and running, and accessible from your application. (if you are going to use pgvector as your vectorstore)

## Usage

### Preparing Your Data

1. **Define the File Path**: Specify the path to your text or PDF file.
2. **Load Embedding Models**: Initialize your embedding model from OpenAI's selection of pre-trained models.

## Clustering and Retrieval

### Initialize the Retrieval System

```python
from Indox import IndoxRetrievalAugmentation
IRA = IndoxRetrievalAugmentation(re_chunk=False)
```

- in the above code, re_chunk is an argument of class IndoxRetrievalAugmentation that specifies if you want to perform re chunking or not. if you enable it, chunking will happen after each summarization, if not, the ckunking will happen only in start of the process. so you also can initialize the code this way:

```python
IRA = IndoxRetrievalAugmentation(re_chunk=True)
```

### Initialize from your own configuration: 

```python
config = {"clustering": {"dim": 10, "threshold": 0.1},
"postgres": {"conn_string": 'postgresql+psycopg2://postgres:xxx@localhost:port/da_name'},
          "qa_model": {"temperature": 0}, "summary_model": {"max_tokens": 100,
"min_len": 30, "model_name": "gpt-3.5-turbo-0125"}, "vector_store": "pgvector", "embedding_model": "openai",
"splitter": "semantic-text-splitter"}

IRA = IndoxRetrievalAugmentation.from_config(config=config, re_chunk=False)
```
**Note**: You need to change postgres config to your postgres credentials if you set vector_store to pgvector

let's examine the config dictionary and its properties:

- clustering
   - dim: Specifies dimension of clustering
   - threshold: Specifies threshold of clustering. if this number is low, more samples will be clustered together. in other words, if you increase this parameter, the number of clusters will increase but size of them will decrease.
- postgres
   - conn_string: Credentials of your postgres database
- qa_model
   - temperature: The temperature of Question Answering model. if this parameter is high, the diversity of the would be high but probability of nonsense and hallucinations would incearse, and if this parameter is low, the diversity of the output would be low but also probability of hallucinations would decrease.
- summary_model
   - max_tokens: Specifies max number of tokens that summary model could generate. more tokens means more cost and also potentially more quality.

   - min_len: Minimum number of tokens that summary model generates.

   - model_name: The model you want to use as your summary model. the defualt is gpt-3.5-turbo-0125. you can use huggingface models that support summarize pipeline, for example, you can use:

```python
  {"summary_model": {"max_tokens":   100, "min_len": 30, "model_name": "Falconsai/medical_summarization"}}
   ```
- vector_store: Specify which vectorstore you want to use. defualt is pgvector, but you also can use "chroma" and "faiss" instead:

```python
{"vector_store": "chroma"}

```
-  embedding_model: The model that is used to embed text. the defualt is openai embeddings, but you can also use "SBert" instead:

```python
{"embedding_model": "SBert"}
```
- splitter: The algorithm used for splitting the text. options are raptor-text-splitter and semantic-text-splitter.

### Generate Chunks

```python
all_chunks = indox.create_chunks_from_document(docs='./sample.txt', max_chunk_size=100)
print("Chunks:", all_chunks)
```
- The max_chunk_size parameter specifies max number of tokens in each chunk.

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
### First, you need to connect to the vectorstore

```python
indox.connect_to_vectorstore(collection_name='sample_c')
```

### Store in PostgreSQL

```python
# you need to set your database credentials in th config.yaml file
indox.store_in_vectorstore(all_chunks)
```


### Querying

Lastly, we can use the IRA and asnwer to queries using answer_question function from IRA object.

```python
response, scores, context = IRA.answer_question(query="How did Cinderella reach her happy ending?", top_k=5)
print("Responses:", response)
print("Retrieve chunks:", context)
print("Scores:", scores)
```
- the top_k argument speficies how many similar documents will be returned from vectorstore.
### Roadmap

- [x] vector stores
   - [x] pgvector
   - [x] chromadb  
   - [x] faiss

- [x] summary models
   - [x] openai chatgpt
   - [x] huggingface models

- [x] embedding models
   - [x] openai embeddings
   - [x] sentence transformer embeddings

- [x] chunking strategies
   - [x] semantic chunking

- [x] add unstructured support

- [x] add simple RAG support
      
- [ ] cleaning pipeline
