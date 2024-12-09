# LanternDB

LanternDB is a document storage solution using PostgreSQL for storing text documents and their vector embeddings. It supports adding documents, performing similarity searches with cosine similarity, and deleting documents from a specified collection.

**Note**: To use LanternDB as the vector store, users need to have a running PostgreSQL instance and provide the connection parameters when initializing the LanternDB vector store.

To use LanternDB as the vector store:

```python
from indoxRag.vector_stores import LanternDB
from indoxRag.embeddings import HuggingFaceEmbedding

db = LanternDB(
    collection_name="your_collection",
    embedding_function=HuggingFaceEmbedding(),
    connection_params={
        'dbname': 'your_db',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'host',
        'port': 'port'
    }
)
```

## Hyperparameters

- **collection_name** [str]: Name of the PostgreSQL table (collection) to use.
- **embedding_function** [Optional[Embeddings]]: Function to generate embeddings for documents.
- **connection_params** [dict]: Connection parameters for PostgreSQL (e.g., dbname, user, password, host, port).
- **dimension** [int]: Dimensionality of vector embeddings (default: 768).
- **text_key** [str]: Key used to store document text in the collection (default: "text").
- **vector_key** [str]: Key used to store document embeddings in the collection (default: "embedding").

## Usage

### Setting Up the Python Environment

### Windows

1. **Create the virtual environment:**

```bash
python -m venv indoxRag
```

2**Activate the virtual environment:**

```bash
indoxRag\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**

```bash
python -m venv indoxRag
```

2. **Activate the virtual environment:**

```bash
source indoxRag/bin/activate
```

### Get Started

**Import HuggingFace API Key**

```python
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
```

**Import Essential Libraries**

```python
from indoxRag import IndoxRetrievalAugmentation
from indoxRag.llms import HuggingFaceModel
from indoxRag.embeddings import HuggingFaceEmbedding
from indoxRag.data_loader_splitter.SimpleLoadAndSplit import SimpleLoadAndSplit
from indoxRag.vector_stores import MongoDB

indoxRag = IndoxRetrievalAugmentation()
mistral_qa = HuggingFaceModel(api_key=HUGGINGFACE_API_KEY,model="mistralai/Mistral-7B-Instruct-v0.2")
embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1",api_key=HUGGINGFACE_API_KEY)
```

**Setting Up Reference Directory and File Path**

```python
from indoxRag.splitter import SemanticTextSplitter

!wget https://raw.githubusercontent.com/osllmai/inDoxRag/master/Demo/sample.txt
file_path = "sample.txt"
with open(file_path, "r") as file:
    text = file.read()
splitter = SemanticTextSplitter(400)
content_chunks = splitter.split_text(text)
```

**Initialize MongoDB**

```python
db = LanternDB(
    collection_name="your_collection",
    embedding_function=HuggingFaceEmbedding(),
    connection_params={
        'dbname': 'your_db',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'localhost',
        'port': 'port'
    }
)

```

**Connecting VectorStore to IndoxRag**

```python
db.add_texts(content_chunks)
```

**Querying the Database**

```python
query = "How cinderella reach her happy ending?"
retriever = indoxRag.QuestionAnswer(vector_database=db, llm=mistral_qa, top_k=5, document_relevancy_filter=True)
answer = retriever.invoke(query=query)
```
