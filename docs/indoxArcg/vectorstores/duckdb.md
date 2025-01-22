# DuckDB

`DuckDB` is a lightweight, in-memory database solution designed for storing text documents and their vector embeddings. It supports adding documents, performing similarity searches using cosine similarity, and managing data within an in-memory DuckDB instance.

**Note**: To use `DuckDB`, users need to have the `duckdb` package installed and provide an embedding function for generating vector embeddings.

To use `DuckDB` as the vector store:

```python
from indoxArcg.vector_stores import DuckDB
from indoxArcg.embeddings import HuggingFaceEmbedding

db = DuckDB(
    embedding_function=HuggingFaceEmbedding(),
    table_name="your_table"
)
```

## Hyperparameters

- **embedding_function** [Any]: Function to generate embeddings for the documents (must be provided).
- **vector_key** [str]: Column name for storing the embeddings in the table (default: "embedding").
- **id_key** [str]: Column name for unique document IDs (default: "id").
- **text_key** [str]: Column name for storing the text of documents (default: "text").
- **table_name** [str]: Name of the DuckDB table to store embeddings (default: "embeddings").

## Usage

### Setting Up the Python Environment

### Windows

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2**Activate the virtual environment:**

```bash
indoxArcg\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
source indoxArcg/bin/activate
```

### Get Started

**Import HuggingFace API Key**

```python
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

**Import Essential Libraries**

```python
from indoxArcg.llms import HuggingFaceModel
from indoxArcg.embeddings import AzureOpenAIEmbeddings
from indoxArcg import IndoxRetrievalAugmentation

indoxArcg = IndoxRetrievalAugmentation()
mistral_qa = HuggingFaceModel(api_key=HUGGINGFACE_API_KEY,model="mistralai/Mistral-7B-Instruct-v0.2")
azure_embed = AzureOpenAIEmbeddings(api_key=OPENAI_API_KEY,model="text-embedding-3-small")
```

**Setting Up Reference Directory and File Path**

```python
from indoxArcg.splitter import RecursiveCharacterTextSplitter
!wget https://raw.githubusercontent.com/osllmai/indoxArcg/master/Demo/sample.txt
file_path = "sample.txt"
with open(file_path, "r") as file:
    text = file.read()
splitter = RecursiveCharacterTextSplitter(400,20)
content_chunks = splitter.split_text(text)
```

**Initialize DuckDB**

```python
from indoxArcg.vector_stores import DuckDB
vector_store = DuckDB(
    embedding_function=azure_embed,
    vector_key="embedding",
    id_key="id",
    text_key="text",
    table_name="embeddings"
)
```

**Connecting VectorStore to indoxArcg**

```python
vector_store.add(texts=content_chunks)
```

**Querying the Database**

```python
query = "How cinderella reach her happy ending?"
retriever = indoxArcg.QuestionAnswer(vector_database=vector_store, llm=mistral_qa, top_k=5, document_relevancy_filter=True)
answer = retriever.invoke(query=query)
```
