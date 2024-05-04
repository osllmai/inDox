# inDox: Advanced Search and Retrieval Augmentation Generative

## Overview
This project, **inDox**, leverages advanced clustering techniques provided by **Raptor** alongside the efficient retrieval capabilities of **pgvector** and other vector stores. It is designed to allow users to interact with and visualize data within a PostgreSQL database effectively. The solution involves segmenting text data into manageable chunks, enhancing retrieval through a custom model, and providing an intuitive interface for querying and retrieving relevant information.

## Prerequisites
Before running this project, ensure that you have the following installed:
- **Python 3.8+**: Required for running the Python backend.
- **PostgreSQL**: Needed if you wish to store your data in a PostgreSQL database.
- **OpenAI API Key**: Necessary if you are using the OpenAI embedding model.

Ensure your system also meets these requirements:
- Access to environmental variables for handling sensitive information like API keys.
- Suitable hardware capable of supporting intensive computational tasks.

## Optional Libraries and Enhancements
### Unstructured Data Handling
For those looking to process unstructured data such as PDF, HTML, Markdown, LaTeX, and plain text files.

### Additional Clustering Layer
- If the `unstructured` library is not used, you can opt to add an extra clustering layer specifically optimized for structured PDFs or text files to enhance data handling.

### PostgreSQL with pgVector
- **Required Version**: Make sure to use PostgreSQL versions that are compatible with `pgvector`. We recommend PostgreSQL 12 or newer to ensure full compatibility with all features.



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

## Config Setup
Before launching your first instance of **inDox**, it's crucial to properly configure the QA model and the embedding model. This configuration is done through the `IRA_3.config` YAML file.

### Initial Configuration
- **Configuration File**: Ensure you locate and modify the `IRA_3.config` YAML file according to your needs before starting the application. This file sets the parameters for the QA and embedding models which are critical for the application’s performance.

### Dynamic Configuration Changes
For changes that need to be applied after the initial setup or during runtime:
- **Modifying Configurations**: Use the following Python snippet to update your settings dynamically:
  ```python
  IRA.config["your_setting_that_need_to_change"] = "new_setting"
  IRA.update_config()


## Clustering and Retrieval

### Initialize the Retrieval System

```python
from Indox import IndoxRetrievalAugmentation
IRA = IndoxRetrievalAugmentation(re_chunk=False)
```

The re_chunk argument in the IndoxRetrievalAugmentation class specifies whether to perform re-chunking of the data:

False: Chunking occurs only at the start of the process.
True: Chunking happens after each summarization process.
```python
IRA = IndoxRetrievalAugmentation(re_chunk=True)
```

## Configuration Details
Here's a breakdown of the config dictionary and its properties:

### Clustering
- `dim`: Specifies the dimension of clustering.
- `threshold`: Lower thresholds mean more samples will be clustered together; higher thresholds increase the number of clusters but decrease their size.

### PostgreSQL
- `conn_string`: Your PostgreSQL database credentials.

### QA Model
- `temperature`: Controls the diversity of the QA model's responses. Higher values increase diversity but also the risk of nonsensical outputs; lower values decrease diversity and reduce risks.

### Summary Model
- `max_tokens`: Maximum token count the summary model can generate.
- `min_len`: Minimum token count the summary model generates.
- `model_name`: Default is `gpt-3.5-turbo-0125`, but it can be replaced with any Hugging Face model supporting the summarization pipeline.

### Embedding Model
- The default embedding model is OpenAI embeddings. Optionally, "SBert" can be used:
  ```python
  {"embedding_model": "SBert"}

### Splitter
Options include `raptor-text-splitter` and `semantic-text-splitter`.

### Considerations for Re-Chunking

- **Using `unstructured` Library**: Setting `re_chunk` to `True` disables the use of the `unstructured` library due to compatibility issues.
- **Extra Clustering Layer**: If `re_chunk` is set to `True` and the user opts for an additional clustering layer, re-chunking is applied to the outputs of the summary model. However, it is crucial to note that if the summary model's output is less than 500 tokens, re-chunking is not recommended due to potential inefficiency and lack of necessity.

### Generate Chunks

```python
documents = IRA.create_chunks(file_path=file_path, unstructured=True)
print("Documents:", documents)
```
- The `max_chunk_size` parameter specifies the maximum number of tokens in each chunk.
- Using the `unstructured` library, users can add files in PDF, HTML, Markdown, LaTeX, or plain text formats. In this scenario, chunking is performed using the `chunk_by_title` method from the `unstructured` library, which organizes the content by titles within the document.


### PostgreSQL Setup with pgvector

If you want to use PostgreSQL for vector storage, you should perform the following steps:

1. **Install pgvector**: To install `pgvector` on your PostgreSQL server, follow the detailed installation instructions available on the official pgvector GitHub repository:
   [pgvector Installation Instructions](https://github.com/pgvector/pgvector)

2. **Add Vector Extension**:
   Connect to your PostgreSQL database and execute the following SQL command to create the `pgvector` extension:

   ```sql
   -- Connect to your database
   psql -U username -d database_name

   -- Run inside your psql terminal
   CREATE EXTENSION vector;
   # Replace the placeholders with your actual PostgreSQL credentials and details


Additionally, for those interested in exploring other vector database options, you can consider using **Chroma** or **Faiss**. These provide alternative approaches to vector storage and retrieval that may better suit specific use cases or performance requirements.

### Next, you need to connect to the vectorstore

```python
indox.connect_to_vectorstore(collection_name='your_collection_name')
```

### Store in vectorstore

```python
# you need to set your database credentials in th config.yaml file
indox.store_in_vectorstore(chunks=documents)
```


### Querying

Lastly, we can use the IRA and asnwer to queries using answer_question function from IRA object.

```python
response = IRA.answer_question(query="your query?!", top_k=5)
print("Responses:", response[0])
print("Retrieve chunks and scores:", response[1])
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
