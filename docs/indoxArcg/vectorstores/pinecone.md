# PineconeVectorStore

Pinecone provides long-term memory for high-performance AI applications. It’s a managed, cloud-native vector database.

**Note**: If you have not created the index before, you can create it through Pinecone and then give the name of the created index as a parameter to the input of PineconeVectoStore. Note that the created index must have dimensions that match the dimensions of the desired embedding model.

To use Pinecone as the vector store, users need to install Pinecone and set the API key, Index_name and Embedding Model.

```python
from indoxArcg.vector_stores import PineconeVectorStore
db = PineconeVectorStore(embedding=embed,
                         pinecone_api_key=PINECONE_API_KEY,
                         index_name=Index_name)
```

#### Hyperparameters

- embedding [Embeddings]: The embedding to be used.
- pinecone_api_key [str]: The API key of Pinecone.
- index_name [str]: The name of the index in the database.

## Usage

### Setting Up the Python Environment

### Windows

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
indoxArcg_judge\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**

```bash
   python3 -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
   source indoxArcg/bin/activate
```

### Get Started

#### Import Essential Libraries

```python
from indoxArcg.llms import HuggingFaceAPIModel
from indoxArcg.embeddings import HuggingFaceEmbedding
from indoxArcg.data_loader_splitter.SimpleLoadAndSplit import SimpleLoadAndSplit
from indoxArcg.vector_stores import PineconeVectorStore

mistral_qa = HuggingFaceAPIModel(api_key=HUGGINGFACE_API_KEY,model="mistralai/Mistral-7B-Instruct-v0.2")
embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1",api_key=HUGGINGFACE_API_KEY)
```

#### Setting Up Reference Directory and File Path

```python
!wget https://raw.githubusercontent.com/osllmai/indoxArcg/master/Demo/sample.txt
file_path = "sample.txt"

simpleLoadAndSplit = SimpleLoadAndSplit(file_path="sample.txt",remove_sword=False,max_chunk_size=200)
docs = simpleLoadAndSplit.load_and_chunk()
```

#### Initialize Pinecone

```python
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
```

#### Create Index and PineconeVectorStore

```python
index_name = "your-index-name"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,      # change the dimension to the desired value
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)
db = PineconeVectorStore(embedding=embed,pinecone_api_key=PINECONE_API_KEY,index_name=index_name)
```

## Usage

Store documents in the vector store:

```python
db.add(docs=docs)
```
