# Categories Overview

This documentation hub explains how **indoxArcg** integrates with various vectorstores, organized by category to simplify selection and setup. Each category serves distinct use cases within the indoxArcg RAG/CAG architecture.

---

### 1. [Purpose-Built Vector Databases](purpose-built-vector-databases.md)

**Optimized for large-scale, high-performance vector workloads in indoxArcg**.  
These databases are designed exclusively for vector operations, enabling fast similarity search and horizontal scalability.

**Supported in indoxArcg**:

- Pinecone
- Milvus
- Qdrant
- Weaviate
- Vespa
- Chroma
- Deep Lake
- Vearch

**Role in indoxArcg**: Ideal for standalone vector indexing and retrieval in production-scale RAG pipelines.  
**Guide**: [Purpose-Built Vector Databases](purpose-built-vector-databases.md)

---

### 2. [General-Purpose Databases with Vector Extensions](general-purpose-vector-databases.md)

**Leverage existing databases for hybrid (vector + traditional) workflows**.  
Integrate vector search into indoxArcg workflows using familiar systems like PostgreSQL or MongoDB.

**Supported in indoxArcg**:

- PostgreSQL (pgvector/**Lantern**)
- Redis (RediSearch)
- MongoDB (Atlas Vector Search)
- Apache Cassandra
- Couchbase
- SingleStore
- DuckDB
- Pathway

**Role in indoxArcg**: Add vector search to transactional data without migrating outside indoxArcg.  
**Guide**: [General-Purpose Databases](general-purpose-vector-databases.md)

---

### 3. [Graph Databases with Vector Support](graph-databases.md)

**Combine graph relationships and vector semantics in indoxArcg**.  
Enhance context-aware retrieval by linking vector embeddings with graph structures.

**Supported in indoxArcg**:

- Neo4jVector
- MemgraphVector

**Role in indoxArcg**: Power complex CAG workflows (e.g., tracing dependencies + semantic context).  
**Guide**: [Graph Databases](graph-databases.md)

---

### 4. [Embedded/Lightweight Libraries](embedded-libraries.md)

**Minimal setup for prototyping or small-scale indoxArcg deployments**.

**Supported in indoxArcg**:

- FAISS

**Role in indoxArcg**: Rapid experimentation with vector search before scaling to production.  
**Guide**: [Embedded Libraries](embedded-libraries.md)

---

## How to Choose for indoxArcg

| Category                       | Best For indoxArcg Use Cases              | Scalability | Hybrid Queries |
| ------------------------------ | ----------------------------------------- | ----------- | -------------- |
| Purpose-Built Vector Databases | Production RAG with massive datasets      | High        | Limited        |
| General-Purpose Databases      | Adding vectors to existing SQL/NoSQL data | Medium      | High           |
| Graph Databases                | Context-aware CAG with relationships      | Medium      | High           |
| Embedded Libraries             | Prototyping or offline analysis           | Low         | None           |

---

## Integration Workflow

1. **Evaluate Needs**: Use the comparison table above to narrow down categories.
2. **Review Guides**: Click category links for setup, configuration, and indoxArcg best practices.
3. **Test**: Use indoxArcg’s `vectorstore_test` module to validate performance.

---

## Troubleshooting

- **Mismatched Latency**: Ensure Purpose-Built Databases are used for latency-sensitive RAG.
- **Hybrid Query Limits**: Graph/General-Purpose Databases require indoxArcg `v2.1+`.
- **Contributing**: Report issues or suggest improvements [here](https://github.com/osllmai/inDox/tree/master/docs/indoxArcg/vectorstores).

---

## Next Steps

Explore category-specific guides:

1. [Purpose-Built Databases](purpose-built-vector-databases.md)
2. [General-Purpose Databases](general-purpose-vector-databases.md)
3. [Graph Databases](graph-databases.md)
4. [Embedded Libraries](embedded-libraries.md)
