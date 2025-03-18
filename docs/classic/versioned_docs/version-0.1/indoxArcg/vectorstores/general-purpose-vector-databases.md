# General-Purpose Databases with Vector Extensions

This guide covers **indoxArcg** integrations with traditional databases enhanced with vector search capabilities.

---

## Supported Vectorstores

### 1. PostgreSQL (pgvector/Lantern)

**Relational database with multiple vector extensions.**

#### pgvector Setup

```bash
pip install pgvector psycopg2
```

#### Lantern Setup

```bash
pip install lantern-postgres
```

#### indoxArcg Integration

```python
# Using pgvector
from indoxArcg.vector_stores import PGVectorStore
pg_db = PGVectorStore(
    host="localhost",
    port=5432,
    dbname="indoxarcg_db",
    user="admin",
    password="secret",
    collection_name="docs",
    embedding=HuggingFaceEmbedding()
)

# Using LanternDB
from indoxArcg.vector_stores import LanternDB
lantern_db = LanternDB(
    collection_name="indoxarcg_docs",
    embedding_function=HuggingFaceEmbedding(),
    connection_params={
        'dbname': 'indoxarcg_db',
        'user': 'admin',
        'password': os.getenv('PG_PASSWORD'),
        'host': 'pg.indoxarcg.com',
        'port': 5432
    }
)
```

#### PostgreSQL Extension Comparison

| Feature         | pgvector | Lantern   |
| --------------- | -------- | --------- |
| Index Type      | IVFFlat  | HNSW      |
| Parallel Builds | Limited  | Optimized |
| Max Dimensions  | 2000     | 16000     |
| Search Speed    | Good     | Excellent |
| indoxArcg Setup | Low      | Medium    |

---

### 2. Redis

**In-memory database with vector search support.**

#### Configuration

```python
from indoxArcg.vector_stores import RedisDB

redis_db = RedisDB(
    host="redis.indoxarcg.com",
    port=6379,
    password=os.getenv('REDIS_PASSWORD'),
    embedding=HuggingFaceEmbedding(),
    prefix="indoxarcg:"
)
```

---

### 3. MongoDB

**Document database with Atlas vector search.**

#### Vector Index Setup

```python
db = MongoDB(
    collection_name="indoxarcg_collection",
    embedding_function=embed_model,
    connection_string="mongodb+srv://user:pass@cluster.indoxarcg.mongodb.net/",
    database_name="vector_db"
)
```

---

### 4. Apache Cassandra

**Distributed NoSQL database with vector support.**

#### Schema Configuration

```python
from indoxArcg.vector_stores import ApacheCassandra

cassandra_db = ApacheCassandra(
    embedding_function=embed_model,
    keyspace="indoxarcg_keyspace"
)
cassandra_db._setup_keyspace()
```

---

### 5. Couchbase

**JSON document store with vector indexing.**

#### Full-Text Search Setup

```python
db = Couchbase(
    embedding_function=embed_model,
    bucket_name="indoxarcg_bucket",
    cluster_url="couchbase://db.indoxarcg.com"
)
```

---

### 6. SingleStore

**Distributed SQL database with vector indexing.**

#### Hybrid Search Configuration

```python
db = SingleStoreVectorDB(
    connection_params={
        'host': 'svc.indoxarcg.com',
        'user': 'admin',
        'password': os.getenv('SINGLESTORE_PWD'),
        'database': 'vector_db'
    },
    embedding_function=embed_model,
    use_vector_index=True
)
```

---

### 7. DuckDB

**Lightweight in-memory OLAP database.**

#### Embedded Usage

```python
from indoxArcg.vector_stores import DuckDB

vector_store = DuckDB(
    embedding_function=embed_model,
    table_name="indoxarcg_embeddings"
)
vector_store.add(texts=chunks)
```

---

### 8. Pathway

**Real-time vector processing engine.**

#### Streaming Setup

```python
from indoxArcg.vector_stores import PathwayVectorClient

client = PathwayVectorClient(
    host="pathway.indoxarcg.com",
    port=8080
)
client.add(docs=real_time_stream)
```

---

## Comparison of General-Purpose Stores

| Database              | Vector Index | Hybrid Queries | Latency | indoxArcg Setup Complexity |
| --------------------- | ------------ | -------------- | ------- | -------------------------- |
| PostgreSQL (pgvector) | IVFFlat      | SQL + Vectors  | Medium  | Low                        |
| PostgreSQL (Lantern)  | HNSW         | SQL + Vectors  | Low     | Medium                     |
| Redis                 | FLAT         | JSON + Vectors | Low     | Low                        |
| MongoDB               | HNSW         | Agg Pipeline   | Medium  | Medium                     |
| Cassandra             | CQL          | CQL + Vectors  | High    | High                       |
| Couchbase             | N1QL         | N1QL + Vectors | Medium  | Medium                     |
| SingleStore           | ANN          | SQL + ANN      | Low     | High                       |
| DuckDB                | ❌           | SQL Only       | Low     | Low                        |
| Pathway               | Streaming    | Real-time      | Ultra   | High                       |

---

## Troubleshooting

### PostgreSQL/Lantern Specific

1. **Extension Not Enabled**

```sql
-- For pgvector
CREATE EXTENSION vector;
-- For Lantern
CREATE EXTENSION lantern;
```

2. **HNSW Index Optimization**

```python
# Lantern-specific index creation
lantern_db.create_index(
    index_type="hnsw",
    metric="cosine",
    m=16,
    ef_construction=64
)
```

3. **Dimension Validation**

```python
# Verify embedding dimensions match
assert len(embed_model.embed_query("test")) == 768, "Mismatch with DB schema"
```

[... Rest of troubleshooting section remains unchanged ...]

---

## Next Steps

[Return to Vectorstore Hub](index.md) | [Graph Databases ➡️](graph-databases.md)

```

```
