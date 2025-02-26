# Graph Databases with Vector Support in indoxArcg

This guide covers **indoxArcg** integrations with graph databases that combine vector search with graph traversal capabilities.

---

## Supported Vectorstores

### 1. MemgraphVector
**Real-time graph database with hybrid vector/keyword search.**

#### Installation
```bash
pip install neo4j
```

#### indoxArcg Integration
```python
from indoxArcg.vector_stores import MemgraphVector

mg_db = MemgraphVector(
    uri="bolt://mg.indoxarcg.com:7687",
    username="admin",
    password=os.getenv('MEMGRAPH_PWD'),
    embedding_function=HuggingFaceEmbedding(),
    search_type='hybrid'  # 'vector' | 'keyword' | 'hybrid'
)

# Hybrid search example
results = mg_db.search(
    query="AI in healthcare",
    search_type="hybrid",
    k=5
)
```

---

### 2. Neo4jVector
**Graph database with native vector indexing.**

#### Setup
```bash
docker run -p 7474:7474 -p 7687:7687 neo4j:5.16.0
```

#### indoxArcg Configuration
```python
from indoxArcg.vector_stores import Neo4jGraph

neo4j_db = Neo4jGraph(
    uri="bolt://neo4j.indoxarcg.com:7687",
    username="neo4j",
    password=os.getenv('NEO4J_PWD')
)

# Store graph documents with entity relationships
neo4j_db.add_graph_documents(
    graph_documents=knowledge_graph,
    base_entity_label=True,
    include_source=True
)

# Query parent-child relationships
relationships = neo4j_db.search_relationships_by_entity(
    entity_id="LLM_Research",
    relationship_type="SUB_FIELD"
)
```

---

## Comparison of Graph Databases

| Feature                | MemgraphVector                  | Neo4jVector                     |
|------------------------|----------------------------------|----------------------------------|
| Search Types           | Vector, Keyword, Hybrid         | Vector + Graph Patterns         |
| Latency                | <50ms (real-time)               | 100-500ms                       |
| indoxArcg Setup        | Medium                          | High                            |
| Relationship Queries   | Basic                           | Cypher Advanced                 |
| Scalability            | High                            | Medium                          |
| Hybrid Search Weights  | Configurable                    | N/A                             |

---

## Key Use Cases
1. **Knowledge Graphs**  
   Connect entities with semantic context:
   ```python
   mg_db.search(query="Blockchain in finance", search_type="hybrid")
   ```

2. **Fraud Detection**  
   Trace suspicious patterns with vector-enhanced relationships:
   ```python
   neo4j_db.search_relationships_by_entity(
       entity_id="Suspicious_Transaction_123",
       relationship_type="LINKED_TO"
   )
   ```

3. **Recommendation Systems**  
   Combine user behavior graphs with content similarity:
   ```python
   mg_db.retrieve(query="Sci-fi movies", search_type="vector", top_k=10)
   ```

---

## Troubleshooting

### Common Issues
1. **Connection Timeouts**  
   Verify graph database status:
   ```bash
   # For Neo4j
   curl http://neo4j.indoxarcg.com:7474

   # For Memgraph
   mg_client = neo4j.GraphDatabase.driver(uri, auth=(user, pwd))
   mg_client.verify_connectivity()
   ```

2. **Missing Relationships**  
   Ensure proper schema initialization:
   ```python
   neo4j_db._init_schema()  # Creates required constraints
   ```

3. **Vector Index Failures**  
   Check embedding dimensions match:
   ```python
   print(len(embed_model.embed_query("test")))  # Should match DB config
   ```

4. **Hybrid Search Imbalance**  
   Adjust weights in Memgraph:
   ```python
   results = mg_db.search(
       query="Quantum computing",
       search_type="hybrid",
       vector_weight=0.7,
       keyword_weight=0.3
   )
   ```

---

## Next Steps
[Return to Vectorstore Hub](../README.md) | [Embedded Libraries ➡️](embedded-libraries.md)
```

---