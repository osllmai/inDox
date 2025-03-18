# Purpose-Built Vector Databases

This guide covers **indoxArcg** integrations with specialized vector databases optimized for high-performance similarity search.

---

## Supported Vectorstores

### 1. Milvus

**Cloud-native vector database for scalable similarity search.**

#### Installation

```bash
# Install Docker
docker --version

# Clone Milvus repo
git clone https://github.com/milvus-io/milvus.git
cd milvus/deployments/docker

# Start Milvus
docker-compose up -d
```

#### indoxArcg Integration

```python
from pymilvus import connections
connections.connect(host='127.0.0.1', port='19530')

from indoxArcg.vector_stores import MilvusVectorStore
db = MilvusVectorStore(collection_name="indoxarcg_collection", embedding=embed)
```

---

### 2. Pinecone

**Managed vector database for production AI applications.**

#### Setup

```python
from pinecone import ServerlessSpec

pc.create_index(
    name="indoxarcg-index",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
```

#### indoxArcg Usage

```python
from indoxArcg.vector_stores import PineconeVectorStore
db = PineconeVectorStore(
    embedding=embed,
    pinecone_api_key=os.getenv('PINECONE_API_KEY'),
    index_name="indoxarcg-index"
)
db.add(docs=docs)
```

---

### 3. Qdrant

**High-performance open-source vector search engine.**

#### Configuration

```python
from indoxArcg.vector_stores import Qdrant

qdrant_db = Qdrant(
    collection_name="indoxarcg_collection",
    embedding_function=HuggingFaceEmbedding(),
    url="https://qdrant-api-url",
    api_key=os.getenv('QDRANT_API_KEY')
)
```

---

### 4. Weaviate

**Semantic search engine with vector+graph hybrid capabilities.**

#### Docker Setup

```bash
docker run -d -p 8080:8080 semitechnologies/weaviate:latest
```

#### indoxArcg Integration

```python
from indoxArcg.vector_stores import WeaviateVectorStore
db = WeaviateVectorStore(
    client=weaviate.Client("http://localhost:8080"),
    index_name="indoxarcg",
    text_key="content"
)
```

---

### 5. Chroma

**Lightweight embedding store for AI applications.**

#### Quick Start

```python
from indoxArcg.vector_stores import ChromaVectorStore
db = ChromaVectorStore(
    collection_name="indoxarcg_docs",
    embedding=HuggingFaceEmbedding()
)
```

---

### 6. DeepLake

**Vector store for deep learning datasets.**

#### Usage

```python
from indoxArcg.vector_stores import Deeplake
db = Deeplake(
    collection_name="indoxarcg_dataset",
    embedding_function=embed_openai_indox
)
db.add(docs=processed_docs)
```

---

### 7. Vearch

**Distributed vector search platform.**

#### Configuration

```python
from indoxArcg.vector_stores import Vearch

db = Vearch(
    embedding_function=embed_model,
    db_name="indoxarcg_db",
    space_name="indoxarcg_space"
)
db.create_space_schema()
```

---

## Comparison of Purpose-Built Stores

| Vectorstore | Scalability | Hybrid Search | Cloud Managed | indoxArcg Setup Complexity |
| ----------- | ----------- | ------------- | ------------- | -------------------------- |
| Milvus      | High        | ✅            | Self-hosted   | Medium                     |
| Pinecone    | Very High   | ❌            | Fully Managed | Low                        |
| Qdrant      | High        | ✅            | Both          | Medium                     |
| Weaviate    | Medium      | ✅ (Graph)    | Both          | High                       |
| Chroma      | Low         | ❌            | Self-hosted   | Low                        |
| DeepLake    | Medium      | ❌            | Managed       | Medium                     |
| Vearch      | High        | ✅            | Self-hosted   | High                       |

---

## Troubleshooting

### Common Issues

1. **Connection Timeouts**

   - Verify ports (`19530` for Milvus, `8080` for Weaviate)
   - Check Docker container status

2. **Dimension Mismatch**  
   Ensure embedding dimensions match index configuration:

   ```python
   print(embed.embed_query("test").shape)  # Should match vectorstore config
   ```

3. **Authentication Failures**  
   Validate API keys using test calls:
   ```python
   # For Pinecone
   import pinecone
   pinecone.list_indexes()  # Should return without errors
   ```

---

## Next Steps

[Return to Vectorstore Hub](index.md) | [General-Purpose Databases ➡️](general-purpose-vector-databases.md)

```

---
```
