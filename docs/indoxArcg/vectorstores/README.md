# Vector Stores in indoxArcg

Vector stores are crucial in Retrieval-Augmented Generation (RAG) and other information retrieval workflows, enabling quick and semantic searches over embeddings. **indoxArcg** provides a variety of integrations with popular open-source and commercial vector databases, offering consistent APIs to:

- **Store** embeddings for text documents, images, or other data.
- **Perform** similarity searches with flexible options (top-k nearest neighbors, hybrid keyword+vector queries, etc.).
- **Retrieve** the best matching results for question-answering or downstream tasks.

Below is a summary of each available vector store in **indoxArcg**:

## 1. [Couchbase Vector](./Couchbase.md)
- **Description**: Integration with [Couchbase](https://www.couchbase.com/), leveraging its flexible JSON document model and built-in Full-Text Search or custom vector indexing features.
- **Use Case**: Ideal if you already rely on Couchbase for distributed caching or document storage and want to add vector similarity on top.

## 2. [Deeplake](./Deeplake.md)
- **Description**: Connects to [Deeplake](https://github.com/activeloopai/deeplake) (by Activeloop), a data lake for deep learning that supports version control of embeddings.
- **Use Case**: If you need a scalable, version-controlled repository for large datasets and embeddings, plus direct GPU-accelerated queries.

## 3. [DuckDB](./duckdb.md)
- **Description**: Backs vector storage with [DuckDB](https://duckdb.org/), an in-process SQL OLAP database engine known for its speed and simplicity.
- **Use Case**: Great for local analytics on embeddings; minimal overhead and easy to set up (single-file usage).

## 4. [Faiss](./Faiss.md)
- **Description**: Interfaces with [Faiss](https://github.com/facebookresearch/faiss), Facebook AI Research’s library for efficient similarity search and clustering of dense vectors.
- **Use Case**: One of the fastest, most popular CPU/GPU-accelerated libraries for large-scale vector search.

## 5. [Lantern](./lantern.md)
- **Description**: Integration with [Lantern](https://docs.lantern.dev/) (if applicable), enabling vector-based data retrieval for specialized analytics.
- **Use Case**: Use if you’re already in the Lantern ecosystem and want to store/serve embeddings within it.

## 6. [MemgraphVector](./memgraphvector.md)
- **Description**: Leverages [Memgraph](https://memgraph.com/) for vector similarity search on graph data. Allows vector, keyword, and hybrid queries.
- **Use Case**: Perfect when you need a **graph database** approach plus semantic search within those graph nodes.

## 7. [Milvus](./milvus.md)
- **Description**: Connects to [Milvus](https://milvus.io/), a purpose-built, cloud-native vector database designed for high-performance vector similarity search.
- **Use Case**: If you have large-scale vector data and require horizontal scalability and advanced indexing techniques.

## 8. [MongoDB Vector](./mongodb.md)
- **Description**: Integration with [MongoDB](https://www.mongodb.com/). Some solutions rely on 2D or multi-dimensional indexes, or custom vector fields.
- **Use Case**: If MongoDB is your main database and you want to store vectors alongside documents without adopting a separate system.

## 9. [Neo4j Graph](./neo4j_graph.md)
- **Description**: Leverages [Neo4j](https://neo4j.com/) as a vector store. Often used with **Neo4jGraph** for entity/relationship storage plus embedding-based similarity queries.
- **Use Case**: Graph-based knowledge modeling plus advanced queries for similarities and relationships in one system.

## 10. [Pathway](./pathway.md)
- **Description**: Integration with [Pathway](https://docs.pathway.com/), a streaming data processing engine that can maintain incremental transformations over vector data.
- **Use Case**: Real-time or streaming-based vector updates and queries.

## 11. [Pinecone](./pinecone.md)
- **Description**: Connects to [Pinecone](https://www.pinecone.io/), a fully managed vector database service with robust scaling and real-time indexing.
- **Use Case**: Cloud-native solution for frictionless vector search; offloads infrastructure management to Pinecone.

## 12. [Postgres Vector](./postgres.md)
- **Description**: Utilizes [Postgres](https://www.postgresql.org/) extensions (like `pgvector`) to store and search embeddings.
- **Use Case**: Works well if your infrastructure already revolves around PostgreSQL and you want minimal overhead or extra components.

## 13. [Qdrant](./qdrant.md)
- **Description**: Integration with [Qdrant](https://qdrant.tech/), an open-source vector similarity search engine. Known for high performance and easy setup.
- **Use Case**: Efficient local or cloud-based vector search with advanced filtering, payload support, and more.

## 14. [Redis Vector](./redis.md)
- **Description**: Uses [Redis](https://redis.io/) with modules like RediSearch or Redis Stack for vector indexing and search.
- **Use Case**: Ideal if you already rely on Redis for caching or data store and want near real-time vector searches.

## 15. [SingleStore Vector](./singlestore.md)
- **Description**: Ties into [SingleStore](https://www.singlestore.com/), a distributed SQL DB with vector functions. Great for real-time analytics on embeddings.
- **Use Case**: Unified solution for analytical queries plus vector search in a single MPP database.

## 16. [Vearch](./vearch.md)
- **Description**: Connects to [Vearch](https://github.com/vearch/vearch), a scalable distributed vector search and analytics engine.
- **Use Case**: Large-scale vector search with cluster-based architecture, suitable for high-throughput applications.

## 17. [Vespa](./Vespa.md)
- **Description**: Interfaces with [Vespa](https://vespa.ai/), a search engine and data processing platform from Yahoo. Provides advanced ranking, vector search, and streaming updates.
- **Use Case**: Enterprise-scale search and recommendation systems needing strong ranking controls and custom query pipelines.

## 18. [Weaviate](./weaviate.md)
- **Description**: Integration with [Weaviate](https://weaviate.io/), an open-source, modular vector database that supports GraphQL-based queries, hybrid search, and more.
- **Use Case**: If you need a schema-based approach with GraphQL queries, built-in classification modules, or external modules for context enrichment.

## 19. [Apache Cassandra](./ApachCassandra.md)
- **Description**: Exploits [Apache Cassandra](https://cassandra.apache.org/) for storing vectors, often combined with specialized indexing approaches or third-party plugins (e.g., stargate + `aiocqlengine`).
- **Use Case**: Highly scalable column-store approach if you already run Cassandra and need to add vector capabilities.

## 20. [Chroma](./chroma.md)
- **Description**: Hooks into [Chroma](https://docs.trychroma.com/), a lightweight open-source vector database.  
- **Use Case**: Simple local or containerized usage with minimal overhead; easy to spin up for prototypes and small-to-medium projects.

---

## Common Concepts & API Patterns

While each vector store integration has unique configuration details, they generally follow a similar structure:

1. **Initialization**  
   ```python
   store = VectorStoreName(
       uri="...",
       username="...",
       password="...",
       embedding_function=my_embedding_function,
       search_type="vector"  # or "keyword"/"hybrid"
   )
   ```

2. **Search**  
   ```python
   results = store.search(query="Some question", search_type="vector", k=5)
   ```
   - **`query`**: The user’s question or text.
   - **`search_type`**: (Optional) if you want to override the default set at initialization.
   - **`k`**: Number of top results.

3. **Retrieve**  
   ```python
   docs, scores = store.retrieve(query="Another query", top_k=3, search_type="keyword")
   ```
   - Returns documents plus their similarity scores.

4. **Closing**  
   ```python
   store.close()
   ```
   - Ensures all connections/resources are properly released.

---

## Choosing the Right Vector Store

Criteria to consider:

- **Hosting/Deployment**: Local, self-hosted, or fully managed cloud?
- **Scalability**: Small prototypes vs. enterprise scale.
- **Data Model**: Hybrid (document + vector), pure vector, or graph-based?
- **Ecosystem**: Does it align with the rest of your stack? (e.g., if you use MongoDB, `mongodb` might be simplest.)
- **Performance**: Does your application require real-time vector updates or occasional batch writes?

---

## Next Steps

1. **Check each store’s documentation** (linked above) for installation/config specifics, example code, and advanced features (e.g., partial updates, custom filtering, etc.).
2. **Set up the store** in your local environment or cloud platform.
3. **Integrate** with **indoxArcg** pipelines, such as `RAG` or custom LLM workflows.
4. **Test** for performance, data consistency, and correctness.

With **indoxArcg**, you can quickly switch between different vector stores by adjusting configuration parameters, letting you prototype in one system (like Chroma or DuckDB) and scale to enterprise solutions (like Pinecone or Weaviate) with minimal code changes.

---

### Additional Resources

- **indoxArcg Documentation**  
  For deeper insights on the `RAG` pipeline, `LLM` classes, or how to create graph-based knowledge structures, see other sections of the [indoxArcg docs](../../).
  
- **Community & Support**  
  Check each vector store’s official docs and community channels (Slack, Discord, GitHub Issues) for advanced configuration help or performance tuning tips.

**Happy Vector Storing!**