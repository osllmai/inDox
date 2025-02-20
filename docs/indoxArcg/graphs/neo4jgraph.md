# Neo4j Integration in indoxArcg

The `Neo4jGraph` class enables you to store, retrieve, and manage Knowledge Graph data (nodes and relationships) in a Neo4j database. This is particularly useful when working with RAG (Retriever Augmented Generation) pipelines, allowing you to persist and query the entities and relationships extracted from text.

## Installation

1. **Install Neo4j Database**  
   Download and install Neo4j Community or Enterprise Edition from the [official website](https://neo4j.com/download/).  
   Once installed, ensure the database is running. By default, it typically listens on the `bolt://localhost:7687` URI.

2. **Install Python driver for Neo4j**  
   ```bash
   pip install neo4j
   ```

3. **Install indoxArcg (if not already installed)**  
   ```bash
   pip install indoxArcg
   ```

## Usage

### 1. Import and Initialization

```python
from indoxArcg.graph import GraphDocument, Node, Relationship
from indoxArcg.core import Document
from indoxArcg.vectorstores import Neo4jGraph

# Instantiate the Neo4jGraph class, providing your Neo4j connection details
graph_store = Neo4jGraph(uri="bolt://localhost:7687", username="neo4j", password="your_password")
```

- **uri**: The Bolt URI for your Neo4j instance.
- **username**: Your Neo4j username (default is usually `"neo4j"`).
- **password**: Your Neo4j password.

### 2. Creating Graph Documents

A `GraphDocument` represents a batch of Nodes and Relationships that you want to store (and optionally the source text they came from).

```python
# Create a source Document
source_document = Document(
    page_content="Queen Elizabeth II reigned from 1952 until her death.",
    metadata={"title": "British Monarchy"}
)

# Create a few nodes
node1 = Node(
    id="queen_elizabeth_ii",
    type="Person",
    text="Queen Elizabeth II"
)
node2 = Node(
    id="crown",
    type="Object",
    text="Crown"
)

# Create a relationship
relationship1 = Relationship(
    source=node1,
    target=node2,
    type="WEARS"
)

# Create a GraphDocument
graph_doc = GraphDocument(
    nodes=[node1, node2],
    relationships=[relationship1],
    source=source_document
)
```

> **Note**:  
> - **Node** objects have the following parameters:
>   - `id`: A unique identifier for the node.
>   - `type`: A label/category to classify the node (e.g., `Person`, `Event`, `Location`).
>   - `embedding` (optional): A list of float values if you are storing vector embeddings.
>   - `text` (optional): Free-form text content.
>   
> - **Relationship** objects link **source** and **target** nodes and define a `type` (e.g., `PARENT_OF`, `MENTORS`, `FOUNDED`).

### 3. Storing the GraphDocument in Neo4j

Once you have one or more `GraphDocument`s, you can add them to your Neo4j database:

```python
graph_store.add_graph_documents(
    graph_documents=[graph_doc],
    base_entity_label=True,
    include_source=True
)
```

- **`base_entity_label`** (bool): Appends a generic label `Entity` to each node (for easier queries across all node types).  
- **`include_source`** (bool): Creates a special `Source` node representing the source `Document`, then links each extracted entity node with a `HAS_SOURCE` relationship.

### 4. Closing the Connection

It’s best practice to close the database session when you’re done:

```python
graph_store.close()
```

### 5. Querying Relationships (Example)

To search for relationships of a given type connected to a specific entity ID, use:

```python
results = graph_store.search_relationships_by_entity(
    entity_id="queen_elizabeth_ii",
    relationship_type="WEARS"
)

for record in results:
    print(record)
```

This will return a list of matched records, each including the source node, the relationship type, and the target node.

---

## Class Reference

### `class Neo4jGraph`
```python
Neo4jGraph(uri: str, username: str, password: str)
```

- **`__init__`**:
  - Initializes the Neo4j driver using the provided `uri`, `username`, and `password`.

- **`add_graph_documents(graph_documents: List[GraphDocument], base_entity_label: bool = True, include_source: bool = True)`**:
  - Adds a list of `GraphDocument` objects to the Neo4j database.
  - If `base_entity_label` is `True`, each node will have an additional label `Entity`.
  - If `include_source` is `True`, each document’s source `Document` is represented as a node labeled `Source`.

- **`search_relationships_by_entity(entity_id: str, relationship_type: str) -> List[Dict]`**:
  - Queries the database for relationships of a certain type emanating from a specific node ID.
  - Returns a list of records with the format `[{"a": nodeA, "rel_type": "RELATIONSHIP", "b": nodeB}, ...]`.

- **`close()`**:
  - Closes the Neo4j driver connection.

### `class GraphDocument`
- Holds lists of `Node` and `Relationship` objects, plus a source `Document`.
- Typically created after processing a chunk of text using the LLM-based extraction.

### `class Node`
- Represents a graph node, with an `id`, a `type`, an optional `embedding` vector, and optional `text`.

### `class Relationship`
- Represents a directed relationship between two nodes, with a `source`, `target`, and `type`.

---

## Best Practices

1. **Indices and Constraints**  
   In Neo4j, consider creating an index on `id` for faster lookups:
   ```cypher
   CREATE INDEX node_id_index FOR (n:Entity) ON (n.id);
   ```
   Similarly, if you’re storing `Source` nodes, you may want an index on their title if you frequently query it.

2. **Batching and Transactions**  
   If you have a very large number of `GraphDocument` objects, you may want to batch writes or use explicit transactions to optimize performance.

3. **Security**  
   Avoid hard-coding credentials. Use environment variables or a configuration manager to store your Neo4j `username` and `password`.

---

## Putting It All Together

```python
from indoxArcg.core import Document
from indoxArcg.graph import GraphDocument, Node, Relationship
from indoxArcg.vectorstores import Neo4jGraph

# 1. Setup Neo4j connection
graph_store = Neo4jGraph(uri="bolt://localhost:7687", username="neo4j", password="password123")

# 2. Build your source document
source_doc = Document(page_content="Some text...", metadata={"title": "Sample Title"})

# 3. Create nodes and relationships
person_node = Node(id="alice", type="Person", text="Alice")
city_node = Node(id="wonderland", type="Location", text="Wonderland")
relationship = Relationship(source=person_node, target=city_node, type="VISITS")

# 4. Create a GraphDocument
graph_doc = GraphDocument(
    nodes=[person_node, city_node],
    relationships=[relationship],
    source=source_doc
)

# 5. Add the graph document to Neo4j
graph_store.add_graph_documents([graph_doc], base_entity_label=True, include_source=True)

# 6. Query relationships
results = graph_store.search_relationships_by_entity(entity_id="alice", relationship_type="VISITS")
print(results)

# 7. Clean up
graph_store.close()
```