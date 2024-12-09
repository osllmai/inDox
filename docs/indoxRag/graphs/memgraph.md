# MemgraphDB

The `MemgraphDB` class allows you to use Memgraph as a graph database to store and query graph-structured data. This can be used in a Retrieval-Augmented Generation (RAG) system or other knowledge graph-related applications.

# Hyperparameters

- **uri** (str): The URI of the Memgraph instance (e.g., "bolt://localhost:7687").
- **username** (Optional[str]): The username for authenticating to the Memgraph database (optional, defaults to an empty string).
- **password** (Optional[str]): The password for authenticating to the Memgraph database (optional, defaults to an empty string).

# Installation

To use Memgraph with this package, you need to install the `neo4j` driver, as it works with Memgraph. You can install it using pip:

```bash
pip install neo4j
```

You will also need a running Memgraph instance. Refer to the official Memgraph installation guide for instructions on setting up a Memgraph instance locally or in the cloud.

# Example Usage

## Initialize the MemgraphDB

You can initialize the `MemgraphDB` class by passing the Memgraph connection details.

```python
from indoxRag.graph import MemgraphDB

memgraph_db = MemgraphDB(uri="bolt://localhost:7687", username="username", password="password")
```

## Add Graph Documents to Memgraph

To add graph documents (which contain nodes and relationships) to the Memgraph database, you can use the `add_graph_documents` method. This method allows you to store a list of graph documents with options to include source metadata and entity labels.

```python
# Assuming `graph_documents` is a list of GraphDocument objects you want to store
memgraph_db.add_graph_documents(graph_documents, base_entity_label=True, include_source=True)
```

- **graph_documents** (List[GraphDocument]): A list of `GraphDocument` objects to store in the Memgraph database.
- **base_entity_label** (bool, optional): If True, adds a base "Entity" label to all nodes. Defaults to True.
- **include_source** (bool, optional): If True, includes the source document as part of the graph. Defaults to True.

## Query Relationships by Entity

To query relationships in Memgraph, you can use the `search_relationships_by_entity` method. This method allows you to search for relationships by specifying the entity ID and the relationship type.

```python
# Search for parent relationships for the entity with ID "Elizabeth_I"
relationships = memgraph_db.search_relationships_by_entity(entity_id="Elizabeth_I", relationship_type="PARENT")

# Print the found relationships
for rel in relationships:
    print(f"{rel['a']['id']} is a {rel['rel_type']} of {rel['b']['id']}")
```

- **entity_id** (str): The ID of the entity for which you want to find relationships.
- **relationship_type** (str): The type of relationship to search for (e.g., "PARENT").

## Close the Connection

After performing your operations, always remember to close the Memgraph connection:

```python
memgraph_db.close()
```

# Example Workflow

Here's an example of how to use `MemgraphDB` to store and query relationships in Memgraph:

```python
from indoxRag.graph import MemgraphDB

# Initialize MemgraphDB
memgraph_db = MemgraphDB(uri="bolt://localhost:7687", username="username", password="password")

# Assuming you have a list of GraphDocument objects ready to store
# graph_documents = List of GraphDocument objects

# Add the graph documents to Memgraph
memgraph_db.add_graph_documents(graph_documents, base_entity_label=True, include_source=True)

# Query relationships of an entity
relationships = memgraph_db.search_relationships_by_entity(entity_id="Elizabeth_I", relationship_type="PARENT")

# Output the relationships
for rel in relationships:
    print(f"{rel['a']['id']} is a {rel['rel_type']} of {rel['b']['id']}")

# Close the Memgraph connection
memgraph_db.close()
```
