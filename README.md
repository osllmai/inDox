# inDox: Advanced Search and Retrieval Augmentation Generative

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

Ensure your PostgreSQL database is up and running, and accessible from your application. (if you are going to use
pgvector as your vectorstore)

## Usage

### Preparing Your Data

1. **Define the File Path**: Specify the path to your text or PDF file.
2. **Load Embedding Models**: Initialize your embedding model from OpenAI's selection of pre-trained models.

## Initial Configuration

- **Configuration File**: Ensure you locate and modify the `IRA.config` YAML file according to your needs before
  starting the application.

## Dynamic Configuration Changes

For changes that need to be applied after the initial setup or during runtime:

- **Modifying Configurations**: Use the following Python snippet to update your settings dynamically:
  ```python
  IRA.config["your_setting_that_need_to_change"] = "new_setting"
  IRA.update_config()

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


