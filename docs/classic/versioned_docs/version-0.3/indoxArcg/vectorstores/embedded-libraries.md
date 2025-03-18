# Embedded/Lightweight Libraries

This guide covers **indoxArcg** integrations with lightweight vector libraries for prototyping and small-scale deployments.

---

## Supported Vectorstores

### FAISS (Facebook AI Similarity Search)

**In-memory vector library for efficient similarity search.**

#### Installation

```bash
pip install faiss-cpu  # or faiss-gpu for CUDA support
```

#### indoxArcg Integration

```python
from indoxArcg.vector_stores import FAISSVectorStore
from indoxArcg.embeddings import HuggingFaceEmbedding

# Initialize with embedding model
embed = HuggingFaceEmbedding(model="all-MiniLM-L6-v2")
db = FAISSVectorStore(embedding=embed)

# Add documents
db.add(docs=chunks)

# Query with RAG pipeline
query = "How does indoxArcg handle document chunking?"
retriever = indoxArcg.QuestionAnswer(
    vector_database=db,
    llm=mistral_qa,
    top_k=5,
    document_relevancy_filter=True
)
answer = retriever.invoke(query=query)
```

---

## Key Features

- **Zero Server Setup**: Runs entirely in memory
- **GPU Acceleration**: Optional CUDA support via `faiss-gpu`
- **Persistent Storage**: Save/load indexes to disk:
  ```python
  db.save_local("faiss_index")  # Save
  db.load_local("faiss_index")  # Load
  ```

---

## When to Use FAISS

1. **Prototyping**: Test RAG workflows locally before cloud deployment
2. **Small Datasets**: &lt;1M embeddings with &lt;=768 dimensions
3. **Cost Sensitivity**: Avoid managed service costs
4. **Offline Requirements**: Air-gapped environments

---

## Limitations

| Aspect         | FAISS Capacity   |
| -------------- | ---------------- |
| Max Embeddings | 1M (RAM-bound)   |
| Persistence    | Manual save/load |
| Hybrid Search  | Not Supported    |
| Scalability    | Single-node only |

---

## Troubleshooting

### Common Issues

1. **Installation Failures**

   - **M1/M2 Macs**: Use conda:
     ```bash
     conda install -c conda-forge faiss-cpu
     ```
   - **CUDA Errors**: Match faiss-gpu version with CUDA toolkit

2. **Dimension Mismatch**

   ```python
   # Verify embedding dimensions
   print(len(embed.embed_query("test")))  # Must match FAISS index
   ```

3. **Memory Errors**
   - Reduce batch size when adding documents:
     ```python
     db.add(docs=chunks, batch_size=100)
     ```

---

## Next Steps

[Return to Vectorstore Hub](index.md) | [Purpose-Built Databases ➡️](purpose-built-vector-databases.md)

```

---
```
