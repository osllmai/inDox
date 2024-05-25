<div style="position: relative; width: 100%; text-align: center;">
<h1>
inDox: Advanced Search and Retrieval Augmentation Generative
</h1>

</div>
<div style="position: relative; width: 100%; text-align: center;">
    <img src="docs/lite-logo 1.png" alt="Logo" style="width: 80%; opacity: 0.7;"/>
</div>

**Indox Retrieval Augmentation** is an innovative application designed to streamline information extraction from a wide
range of document types, including text files, PDF, HTML, Markdown, and LaTeX. Whether structured or unstructured, Indox
provides users with a powerful toolset to efficiently extract relevant data.

Indox Retrieval Augmentation is an innovative application designed to streamline information extraction from a wide
range of document types, including text files, PDF, HTML, Markdown, and LaTeX. Whether structured or unstructured, Indox
provides users with a powerful toolset to efficiently extract relevant data. One of its key features is the ability to
intelligently cluster primary chunks to form more robust groupings, enhancing the quality and relevance of the extracted
information.
With a focus on adaptability and user-centric design, Indox aims to deliver future-ready functionality with more
features planned for upcoming releases. Join us in exploring how Indox can revolutionize your document processing
workflow, bringing clarity and organization to your data retrieval needs.

## Prerequisites

Before running this project, ensure that you have the following installed:

- **Python 3.8+**: Required for running the Python backend.
- **PostgreSQL**: Needed if you wish to store your data in a PostgreSQL database.
- **OpenAI API Key**: Necessary if you are using the OpenAI embedding model.
- **HuggingFace API Key**: Necessary if you are using the HuggingFace llms.

Ensure your system also meets these requirements:

- Access to environmental variables for handling sensitive information like API keys.
- Suitable hardware capable of supporting intensive computational tasks.

## Installation


## Getting Started

The following command will install the latest stable inDox

```
pip install Indox
```

To install the latest development version, you may run

```
pip install git+https://github.com/osllmai/inDox@main
```

To configure the CLI, run

```
indox configure
```


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

Set your `OPENAI_API_KEY` or `HF_API_KEY` in your environment variables for secure access.

### Database Setup

Ensure your PostgreSQL database is up and running, and accessible from your application. This is necessary if you plan to use pgvector as your vector store.

Alternatively, you can use Chroma or Faiss as your vector store. Make sure to specify your choice and the necessary configurations in the config.yaml file.

## Usage

### Preparing Your Data

1. **Define the File Path**: Specify the path to your text or PDF file.
2. **Load Embedding Models**: Initialize your embedding model from OpenAI's selection of pre-trained models.

# Quick Start 

## Import Indox Package

Import the necessary classes from the Indox package.

``` python
from Indox import IndoxRetrievalAugmentation
```

### Initialize Indox

Create an instance of IndoxRetrievalAugmentation.

``` python
Indox = IndoxRetrievalAugmentation()
```
## Initial Configuration

- **Configuration File**: Ensure you locate and modify the `Indox.config` YAML file according to your needs before
  starting the application.

## Dynamic Configuration Changes

For changes that need to be applied after the initial setup or during runtime:

- **Modifying Configurations**: Use the following Python snippet to update your settings dynamically:
  ```python
  Indox.config["your_setting_that_need_to_change"] = "new_setting"
  Indox.update_config()

## Configuration Details

Here's a breakdown of the config dictionary and its properties:

### PostgreSQL

- `conn_string`: Your PostgreSQL database credentials.

### Summary Model

- `max_tokens`: Maximum token count the summary model can generate.
- `min_len`: Minimum token count the summary model generates.
- `model_name`: Default is `gpt-3.5-turbo-0125`, but it can be replaced with any Hugging Face model supporting the
  summarization pipeline.

### PostgreSQL Setup with pgvector

If you want to use PostgreSQL for vector storage, you should perform the following steps:

1. **Install pgvector**: To install `pgvector` on your PostgreSQL server, follow the detailed installation instructions
   available on the official pgvector GitHub repository:
   [pgvector Installation Instructions](https://github.com/pgvector/pgvector)

2. **Add Vector Extension**:
   Connect to your PostgreSQL database and execute the following SQL command to create the `pgvector` extension:

   ```sql
   -- Connect to your database
   psql -U username -d database_name

   -- Run inside your psql terminal
   CREATE EXTENSION vector;
   # Replace the placeholders with your actual PostgreSQL credentials and details

Additionally, for those interested in exploring other vector database options, you can consider using **Chroma** or *
*Faiss**. These provide alternative approaches to vector storage and retrieval that may better suit specific use cases
or performance requirements.

### Importing QA and Embedding Models

``` python
from Indox.QaModels import OpenAiQA
```

``` python
from Indox.Embeddings import OpenAiEmbedding
```


``` python
openai_qa = OpenAiQA(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-0125")
openai_embeddings = OpenAiEmbedding(model="text-embedding-3-small",openai_api_key=OPENAI_API_KEY)
```

## Modifying Configuration Settings

To change a configuration setting, you can directly modify the
`Indox.config` dictionary. Here is an example of how you can update a
configuration setting:

``` python
# Example of modifying a configuration setting
Indox.config["old_config"] = "new_config"

# Applying the updated configuration
Indox.update_config()
```


We take advantage of the `unstructured` library to load
documents and split them into chunks by title. This method helps in
organizing thme document into manageable sections for further
processing.

``` python
from Indox.DataLoaderSplitter import UnstructuredLoadAndSplit
```

``` python
docs_unstructured = UnstructuredLoadAndSplit(file_path=file_path)
```

    Starting processing...
    End Chunking process.

Storing document chunks in a vector store is crucial for enabling
efficient retrieval and search operations. By converting text data into
vector representations and storing them in a vector store, you can
perform rapid similarity searches and other vector-based operations.

``` python
Indox.connect_to_vectorstore(collection_name="sample",embeddings=openai_embeddings)
Indox.store_in_vectorstore(chunks=docs_unstructured)
```

## Quering

``` python
query = "your query!!??"
```

``` python
response_openai = Indox.answer_question(query=query,qa_model=openai_qa)
```

``` python
answer = response_openai[0]
```

``` python
context, score = response_openai[1]
```

