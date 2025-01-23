# MongoDB

MongoDB is a document-oriented NoSQL database that provides high performance, high availability, and easy scalability. This implementation uses MongoDB for document storage and retrieval, supporting vector similarity search using cosine similarity when an embedding function is provided.

**Note**: To use MongoDB as the vector store, users need to install pymongo and have a running MongoDB instance. The MongoDB connection string, database name, and collection name should be provided when initializing the MongoDB vector store.

To use MongoDB as the vector store:

```python
from indoxArcg.vector_stores import MongoDB
from indoxArcg.embeddings import HuggingFaceEmbedding

db = MongoDB(
    collection_name="your_collection",
    embedding_function=HuggingFaceEmbedding(),
    connection_string="mongodb://localhost:27017/",
    database_name="vector_db"
)
```

## Hyperparameters

- **collection_name** [str]: Name of the MongoDB collection to use.
- **embedding_function** [Optional[Embeddings]]: Function to generate embeddings for documents.
- **connection_string** [str]: MongoDB connection string (default: "mongodb://localhost:27017/").
- **database_name** [str]: Name of the MongoDB database to use (default: "vector_db").
- **index_name** [str]: Name of the index used for similarity search (default: "default").
- **text_key** [str]: Key used to store document text in the collection (default: "text").
- **embedding_key** [str]: Key used to store document embeddings in the collection (default: "embedding").

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
```

**Import Essential Libraries**

```python
from indoxArcg.llms import HuggingFaceAPIModel
from indoxArcg.embeddings import HuggingFaceEmbedding
from indoxArcg.data_loader_splitter.SimpleLoadAndSplit import SimpleLoadAndSplit
from indoxArcg.vector_stores import MongoDB

mistral_qa = HuggingFaceAPIModel(api_key=HUGGINGFACE_API_KEY,model="mistralai/Mistral-7B-Instruct-v0.2")
embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1",api_key=HUGGINGFACE_API_KEY)
```

**Setting Up Reference Directory and File Path**

```python
!wget https://raw.githubusercontent.com/osllmai/indoxArcg/master/Demo/sample.txt
file_path = "sample.txt"
simpleLoadAndSplit = SimpleLoadAndSplit(file_path=file_path, remove_sword=False, max_chunk_size=200)
docs = simpleLoadAndSplit.load_and_chunk()
```

**Initialize MongoDB**

```python
db = MongoDB(
    collection_name="indoxArcg_collection",
    embedding_function=embed,
    connection_string="mongodb://localhost:27017/",
    database_name="vector_db"
)
```

**Connecting VectorStore to indoxArcg**

```python
db.add(docs=docs)
```
